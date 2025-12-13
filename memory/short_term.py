from typing import List, Dict, Optional
from utils.embed import get_embedding, cosine_similarity, find_relevant_topics
from utils.file_utils import get_next_session_num
from utils.openai_base import generate_with_retry
from utils.model_call_utils import close_topic
from utils.prompts import topic_switch_decision_prompt
from utils.models import TopicSwitchDecision
from pathlib import Path
from datetime import datetime
from concurrent.futures import Future
import json
import uuid
from memory.topics import Topic

# Approximate tokens: ~4 chars per token, so 8000 tokens ≈ 32k chars
# Using 25k chars as safe limit for context window
MAX_CONTEXT_CHARS = 25000

class ShortTermMemory:
    """manages conversation topics and messages for a single session"""
    
    def __init__(self, username: str, session_num: Optional[int] = None):
        self.username = username
        self.base_path = Path.cwd()
        
        # session directory: sessions/{username}.{session_num}/
        if session_num is None:
            session_num = get_next_session_num(self.username)
        self.session_num = session_num
        self.session_dir = self.base_path / "sessions" / f"{username}.{session_num}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # session files (for debugging, not used in normal flow)
        self.topics_file = self.session_dir / "topics.json"
        self.embeddings_file = self.session_dir / "topics.embeddings"
        
        # in-memory topics for this session
        self.topics: List[Topic] = []
        self.curr_open_topic: Optional[Topic] = None  # current open topic
        self.current_topic_id: Optional[str] = None
        self.topic_embeddings: Dict[str, List[float]] = {}  # topic_id -> embedding
        self.pending_summary_future: Optional[Future] = None  # future for topic summary gen
        
        # don't create topic at init - wait for first message
    
    
    def get_context(self, query: str) -> dict:
        """return recent messages plus summaries and retrieved past topics"""
        if not self.curr_open_topic:
            return {
                "recent_messages": [],
                "closed_topics_summaries": [],
                "relevant_topics": [],
            }
        
        # 1. messages from current open topic
        recent_messages = self.curr_open_topic.messages
        
        # 2. summaries of all closed topics in this session
        closed_topics_summaries = [
            {
                "id": t.id,
                "name": t.name,
                "summary": t.summary,
            }
            for t in self.topics
        ]
        
        # 3. semantic search over closed topics
        relevant_topics: List[Dict] = []
        if self.topics and query.strip():
            relevant_topic_objs = find_relevant_topics(
                query,
                self.topics,
                self.topic_embeddings,
                max_k=2,
                min_threshold=0.75,
            )
            
            for topic in relevant_topic_objs:
                relevant_topics.append(
                    {
                        "id": topic.id,
                        "name": topic.name,
                        "summary": topic.summary,
                        "messages": topic.messages,
                    }
                )
        
        return {
            "recent_messages": recent_messages,
            "closed_topics_summaries": closed_topics_summaries,
            "relevant_topics": relevant_topics,
        }
    
    def add_single_turn(self, user_message: str, assistant_response: str, executor=None) -> Optional[Future]:
        """handle topic shift for each turn
        
        1. save to curr topic
        2. check for topic shift -> if so, close prev topic and start new one
        3. check for context window limit -> if so, close topic and start new one
        """
        if not self.curr_open_topic:
            self._start_new_topic()
        
        timestamp = datetime.utcnow().isoformat()
        
        # 1. save to current topic
        self.curr_open_topic.add_message_to_topic({
            'role': 'user',
            'content': user_message,
            'timestamp': timestamp
        })
        
        self.curr_open_topic.add_message_to_topic({
            'role': 'assistant',
            'content': assistant_response,
            'timestamp': timestamp
        })
        
        # 2. check for topic shift
        if self._detect_topic_shift(user_message):
            # remove last turn from current topic before closing (belongs to new topic)
            if len(self.curr_open_topic.messages) >= 2:
                self.curr_open_topic.messages = self.curr_open_topic.messages[:-2]
            
            summary_future = self._close_current_topic(executor=executor)
            
            # start new topic and add the turn that triggered the switch
            self._start_new_topic()
            self.curr_open_topic.add_message_to_topic({
                'role': 'user',
                'content': user_message,
                'timestamp': timestamp
            })
            self.curr_open_topic.add_message_to_topic({
                'role': 'assistant',
                'content': assistant_response,
                'timestamp': timestamp
            })
            return summary_future
        
        # 3. check for context window limit
        if self._check_context_window_limit():
            summary_future = self._close_current_topic(executor=executor)
            self._start_new_topic()
            return summary_future
        
        return None
    
    def _check_context_window_limit(self) -> bool:
        """check if current topic hit the context window limit"""
        if not self.curr_open_topic:
            return False
        
        # estimate total chars in messages
        total_chars = sum(len(msg.get('content', '')) for msg in self.curr_open_topic.messages)
        
        return total_chars > MAX_CONTEXT_CHARS
    
    def _detect_topic_shift(self, message: str) -> bool:
        """two-stage topic shift detection
        
        1. fast check: embedding similarity (if > 0.45, no shift)
        2. if similarity <= 0.45, verify with llm to avoid false positives
        """
        if not self.curr_open_topic or len(self.curr_open_topic.messages) < 3:
            return False
        
        # stage 1: embedding similarity check (fast, cheap)
        recent_messages = self.curr_open_topic.messages[-3:]
        context_text = " ".join([m.get('content', '') for m in recent_messages])
        
        current_emb = get_embedding(message)
        context_emb = get_embedding(context_text)
        similarity = cosine_similarity(current_emb, context_emb)
        
        # high similarity = same topic, skip llm check
        if similarity > 0.45:
            return False
        
        # stage 2: low similarity - verify with llm
        return self._verify_topic_switch_with_llm(message, recent_messages)
    
    def _verify_topic_switch_with_llm(self, current_message: str, recent_messages: List[Dict]) -> bool:
        """ask llm if this is a new topic"""
        recent_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in recent_messages[-5:]
        ])
        
        prompt = topic_switch_decision_prompt(recent_text, current_message)
        try:
            result = generate_with_retry(prompt, TopicSwitchDecision)
        except Exception:
            return False
        
        if result.switch:
            return True
        
        return False
    
    def _close_current_topic(self, executor=None) -> Optional[Future]:
        """close current topic, add to topics list, return future for summary gen
        
        if executor provided, summary gen runs in parallel
        returns None if no topic to close, or future that completes with closed topic
        """
        if not self.curr_open_topic:
            return None
        
        topic_id = self.curr_open_topic.id
        closed_at = datetime.utcnow().isoformat()
        self.curr_open_topic.closed_at = closed_at
        
        # skip only if topic is completely empty (0 messages)
        if not self.curr_open_topic.messages:
            self.curr_open_topic = None
            self.current_topic_id = None
            return None
        
        # for topics with 1 message, skip llm analysis but still persist
        if len(self.curr_open_topic.messages) < 1:
            self.curr_open_topic.name = self.curr_open_topic.name or "Empty Topic"
            self.curr_open_topic.summary = "Brief exchange with minimal content."
            
            # generate embedding immediately for short topics
            summary_text = self.curr_open_topic.name or "Unnamed Topic"
            self.topic_embeddings[topic_id] = get_embedding(summary_text)
            
            # keep topic in in-memory list
            closed_topic = self.curr_open_topic
            self.topics.append(closed_topic)
            
            # clear current topic
            self.curr_open_topic = None
            self.current_topic_id = None
            
            return None  # no async work needed
        
        # prepare messages for summary generation
        messages_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in self.curr_open_topic.messages
        ])
        
        # keep reference to topic before clearing
        closed_topic = self.curr_open_topic
        
        # clear current topic immediately
        self.curr_open_topic = None
        self.current_topic_id = None
        
        # add topic to list immediately (summary updated when future completes)
        self.topics.append(closed_topic)
        
        if executor:
            # generate summary in parallel
            future = executor.submit(self._generate_and_update_topic_summary, closed_topic, messages_text, topic_id)
            self.pending_summary_future = future
            return future
        else:
            # generate summary synchronously
            self._generate_and_update_topic_summary(closed_topic, messages_text, topic_id)
            return None
    
    def _generate_and_update_topic_summary(self, topic: Topic, messages_text: str, topic_id: str) -> Topic:
        """generate topic summary and update topic object, then generate embedding"""
        try:
            result = close_topic(messages_text)
            topic.name = result.label
            topic.summary = result.summary
        except Exception:
            if not topic.name:
                topic.name = "Unnamed Topic"
            if not topic.summary:
                topic.summary = "No summary available"
        
        # generate embedding now that we have the summary
        summary = topic.summary or ""
        if summary and summary not in ["No summary available", "Brief exchange with minimal content."]:
            summary_text = summary
        else:
            summary_text = topic.name or "Unnamed Topic"
        self.topic_embeddings[topic_id] = get_embedding(summary_text)
        
        return topic
    
    def _start_new_topic(self):
        """start a new topic (name generated when topic closes)"""
        topic_id = str(uuid.uuid4())
        self.current_topic_id = topic_id
        
        self.curr_open_topic = Topic(
            id=topic_id,
            name="",  # Empty initially, will be generated alongside summary when topic closes
            summary=None,
            messages=[],
            created_at=datetime.utcnow().isoformat(),
            closed_at=None
        )
    
    def _topic_to_dict(self, topic: Topic) -> Dict:
        """Convert Topic object to dict matching schema"""
        return {
            'id': topic.id,
            'name': topic.name,
            'summary': topic.summary,
            'messages': topic.messages,
            'created_at': topic.created_at,
            'closed_at': topic.closed_at
        }
    
    def _dict_to_topic(self, topic_dict: Dict) -> Optional[Topic]:
        """Convert dict to Topic object"""
        try:
            return Topic(
                id=topic_dict.get('id', ''),
                name=topic_dict.get('name', 'Unnamed'),
                summary=topic_dict.get('summary'),
                messages=topic_dict.get('messages', []),
                created_at=topic_dict.get('created_at', datetime.utcnow().isoformat()),
                closed_at=topic_dict.get('closed_at')
            )
        except Exception:
            return None
    
    def get_all_topics(self) -> List[Dict]:
        """Get all topics (closed + current open) as dicts for aggregation/serialization."""
        all_topics = [self._topic_to_dict(t) for t in self.topics]
        if self.curr_open_topic:
            all_topics.append(self._topic_to_dict(self.curr_open_topic))
        return all_topics
    
    def save(self):
        """Save topics to JSON"""
        # Convert current open topic to dict if it exists
        all_topics = self.topics.copy()
        if self.curr_open_topic:
            all_topics.append(self._topic_to_dict(self.curr_open_topic))
        
        data = {
            'topics': all_topics,
            'current_topic_id': self.current_topic_id
        }
        
        # Note: We validate topics when loading, but skip validation here for performance
        # Full validation happens in load_topics()
        
        with open(self.topics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_embeddings(self):
        """Save topic embeddings to JSON"""
        with open(self.embeddings_file, 'w') as f:
            json.dump(self.topic_embeddings, f, indent=2)
    