from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, Future

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.topics import Topic
from utils.models import FactsResponse

class MemoryManager:
    """coordinates short-term and long-term memory for one user"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.long_term = LongTermMemory(user_id)
        self.short_term = ShortTermMemory(user_id)

    def get_context(self, query: str) -> Dict:
        """build combined context from both memory layers"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            lt_future = executor.submit(self.long_term.get_context, query)
            st_future = executor.submit(self.short_term.get_context, query)
            long_term_ctx = lt_future.result()
            short_term_ctx = st_future.result()
        
        return {"long_term": long_term_ctx, "short_term": short_term_ctx}

    def add_single_turn(self, user_message: str, assistant_response: str, executor=None) -> Optional[Future]:
        """Record turn in short-term memory, persist any closed topic to long-term
        
        returns future if topic closed and summary gen running in parallel
        caller should wait for future before next query, then call _persist_closed_topic
        
        if None returned, either no topic closed, or closed synchronously
        for sync case, check short_term.topics for newly closed topics
        """
        topics_before = len(self.short_term.topics)
        summary_future = self.short_term.add_single_turn(user_message, assistant_response, executor)
        topics_after = len(self.short_term.topics)
        
        if summary_future is not None:
            # topic closed, summary gen in parallel
            # persistence happens in agent.py after future completes
            return summary_future
        
        # check if topic closed synchronously (no future, but topics list grew)
        if topics_after > topics_before:
            # topic closed synchronously, persist immediately
            closed_topic = self.short_term.topics[-1]
            self._persist_closed_topic(closed_topic)
        
        return None

    def save_facts(self, facts: FactsResponse) -> None:
        """persist extracted facts into long-term profile"""
        self.long_term.save_facts_to_longterm(facts)

    def end_session(self) -> None:
        """close any open topic and update notepad"""
        # close open topic and persist if needed
        if self.short_term.curr_open_topic:
            topics_before = len(self.short_term.topics)
            # close topic (summary gen sync since no executor)
            self.short_term._close_current_topic(executor=None)
            topics_after = len(self.short_term.topics)
            
            # if topic closed, persist it
            if topics_after > topics_before:
                closed_topic = self.short_term.topics[-1]
                self._persist_closed_topic(closed_topic)

        # update notepad
        self.long_term.update_notepad()

    def _persist_closed_topic(self, topic: Topic) -> None:
        """persist a single closed topic into long-term memory"""
        # skip only completely empty topics (0 messages)
        if not topic.messages:
            return
        
        emb = self.short_term.topic_embeddings.get(topic.id)
        topic_dict = {
            "id": topic.id,
            "name": topic.name,
            "summary": topic.summary,
            "messages": topic.messages,
            "created_at": topic.created_at,
            "closed_at": topic.closed_at,
        }
        self.long_term.save_all_session_topics(
            [topic_dict],
            {topic.id: emb} if emb is not None else {},
        )


