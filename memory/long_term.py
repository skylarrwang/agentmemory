import json
from typing import List, Dict
from pathlib import Path
from datetime import datetime

import numpy as np

from memory.topics import Topic
from utils.embed import get_embedding, find_relevant_topics
from utils.openai_base import generate_with_retry
from utils.models import NotepadResponse, FactsResponse
from utils.prompts import compress_notepad_prompt

class LongTermMemory:
    """Stores user profile, notes, and topics across all sessions"""
    
    def __init__(self, username: str):
        self.username = username
        self.base_path = Path.cwd()
        
        # long-term memory directory: longterm_memory/{username}/
        self.lt_dir = self.base_path / "longterm_memory" / username
        self.lt_dir.mkdir(parents=True, exist_ok=True)
        
        # filepaths
        self.facts_file = self.lt_dir / "facts.json"
        self.notepad_file = self.lt_dir / "notepad.md"
        self.all_session_topics_file = self.lt_dir / "all_session_topics.json"
        self.all_session_topics_embeddings_file = self.lt_dir / "all_session_topics.embeddings"
        
        # populated in .load below
        self.facts: Dict = {}  # profile facts (kv pairs)
        self.notepad: str | None = None
        self.all_session_topics: List[Topic] = []
        self.all_topic_embeddings: Dict[str, np.ndarray] = {}
        
        # populate in-memory
        self.load()
    
    def get_context(self, query: str, k: int = 3) -> dict:
        """
        Builds long term context:
        
        Always include:
        1. KV facts 
        2. Notepad
        3. Summary from globally saved topics (top k retrieved from all_session_topics)
        """
        # 1. KV facts
        facts = [f"{field}: {data['value']}" for field, data in self.facts.items()]
        
        # 2. Notepad
        notepad = self._read_notepad()
        # limit notepad length to avoid token bloat
        if len(notepad) > 2000:
            notepad = notepad[:2000] + "..."
        
        # 3. Retrieval over relevant topics from all sessions
        relevant_topics: List[Dict] = []
        if self.all_session_topics and self.all_topic_embeddings and query.strip():
            relevant_topic_objs = find_relevant_topics(
                query,
                self.all_session_topics,
                self.all_topic_embeddings,
                max_k=k,
                min_threshold=0.5,
            )
            
            for topic in relevant_topic_objs:
                relevant_topics.append(
                    {
                        "id": topic.id,
                        "name": topic.name,
                        "summary": topic.summary,
                        "messages": topic.messages,
                        "created_at": topic.created_at,
                        "closed_at": topic.closed_at,
                    }
                )
        
        return {
            "facts": facts,
            "notepad": notepad,
            "relevant_topics": relevant_topics,
        }
    
    def save_facts_to_longterm(self, facts: FactsResponse):
        """store extracted facts in facts.json with importance and timestamps"""
        if not facts or not facts.facts:
            return
        
        now = datetime.utcnow().isoformat()
        
        for fact in facts.facts:
            value_str = fact.value.strip()
            if not value_str:
                continue
            importance = int(fact.importance)
            importance = max(1, min(10, importance))
            
            self.facts[fact.field] = {
                "value": value_str,
                "importance": importance,
                "updated_at": now,
            }
        
        # optionally prune stale low-importance facts
        self._prune_facts()
        
        # persist all facts
        with self.facts_file.open("w", encoding="utf-8") as f:
            json.dump(self.facts, f, indent=2)
    
    def _read_notepad(self) -> str:
        """read notepad file"""
        if self.notepad_file.exists():
            return self.notepad_file.read_text(encoding='utf-8')
        return ""
    
    def update_notepad(self):
        """reflect on session and update notepad only if there are essential strategic insights"""
        # get session topics to provide context
        session_topics = self.all_session_topics[-10:] if len(self.all_session_topics) > 10 else self.all_session_topics
        
        # filter to only topics with actual messages
        topics_with_messages = [t for t in session_topics if t.messages and len(t.messages) >= 2]
        
        # skip notepad update if there are no topics with messages
        if not topics_with_messages:
            return
        
        current_notepad = self._read_notepad()
        
        session_summary = "\n".join([
            f"- {t.name}: {t.summary}" for t in topics_with_messages[-5:]
        ]) if topics_with_messages else "(no topics yet)"
        
        prompt = f"""You maintain a notepad of ESSENTIAL strategic insights about how to best interact with this user.

The notepad should ONLY contain:
- Important conversation patterns or preferences (e.g. "User prefers direct answers", "Responds well to examples")
- Communication strategies that work well (e.g. "User values brevity", "Appreciates technical depth")
- Significant relationship context that affects future interactions
- Core values or principles that guide the user's decisions

DO NOT include:
- Casual conversation topics or temporary interests
- Facts already captured in the user profile
- Trivial observations
- Session summaries or topic recaps

If this session was routine/ordinary and didn't reveal any important strategic insights, return the current notepad unchanged.

Current notepad:
{current_notepad if current_notepad else "(empty)"}

Recent session topics:
{session_summary}

Return JSON in this exact format. The "updated_notepad" field must be a plain text string, NOT a structured object:
{{
  "updated_notepad": "plain text string here with strategic insights, or current notepad if nothing to add"
}}

IMPORTANT: updated_notepad must be a string value, not a JSON object or array.
"""
        try:
            response = generate_with_retry(prompt, NotepadResponse)
            self.notepad = response.updated_notepad
        except ValueError as e:
            # if parsing fails, log error and keep current notepad
            print(f"warning: failed to parse notepad update, keeping current notepad: {e}")
            return
        
        self._maybe_compress_notepad()
        self.save_notepad()
    
    def save_notepad(self):
        """save notepad to file"""
        self.notepad_file.write_text(self.notepad or "", encoding="utf-8")
    
    def save_all_session_topics(self, topics: List[Dict], embeddings: Dict[str, object]):
        """merge and persist session topics and their embeddings into long-term memory"""
        if not topics:
            return
        
        # convert incoming dicts to topic objects and merge
        for t in topics:
            topic_obj = Topic(
                id=t.get("id"),
                name=t.get("name"),
                summary=t.get("summary"),
                messages=t.get("messages", []),
                created_at=t.get("created_at"),
                closed_at=t.get("closed_at"),
            )
            self.all_session_topics.append(topic_obj)
            
            topic_id = topic_obj.id
            if not topic_id:
                continue
            emb = embeddings.get(topic_id)
            if emb is None:
                # use summary if meaningful, otherwise fall back to name
                summary = topic_obj.summary or ""
                if summary and summary not in ["No summary available", "Brief exchange with minimal content."]:
                    summary_text = summary
                else:
                    summary_text = topic_obj.name or "Unnamed Topic"
                if not summary_text.strip():
                    continue
                emb = get_embedding(summary_text)
            self.all_topic_embeddings[topic_id] = emb
        
        # persist topics as list[dict]
        serialized_topics = [
            {
                "id": t.id,
                "name": t.name,
                "summary": t.summary,
                "messages": t.messages,
                "created_at": t.created_at,
                "closed_at": t.closed_at,
            }
            for t in self.all_session_topics
        ]
        with self.all_session_topics_file.open("w", encoding="utf-8") as f:
            json.dump(serialized_topics, f, indent=2)
        
        # persist embeddings as id -> list[float]
        serializable_embeddings = {
            topic_id: (emb.tolist() if hasattr(emb, "tolist") else emb)
            for topic_id, emb in self.all_topic_embeddings.items()
        }
        with self.all_session_topics_embeddings_file.open("w", encoding="utf-8") as f:
            json.dump(serializable_embeddings, f, indent=2)
    
    def load(self):
        """load profile, notepad, and all session topics from disk into memory"""
        # load facts
        if self.facts_file.exists():
            with self.facts_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.facts = data
        
        # load all session topics
        if self.all_session_topics_file.exists():
            with self.all_session_topics_file.open("r", encoding="utf-8") as f:
                all_session_topics_json = json.load(f)
            for t in all_session_topics_json:
                topic_obj = Topic(
                    id=t.get("id"),
                    name=t.get("name"),
                    summary=t.get("summary"),
                    messages=t.get("messages", []),
                    created_at=t.get("created_at"),
                    closed_at=t.get("closed_at"),
                )
                self.all_session_topics.append(topic_obj)
        
        # load all topic embeddings
        if self.all_session_topics_embeddings_file.exists():
            with self.all_session_topics_embeddings_file.open("r", encoding="utf-8") as f:
                stored_embeddings = json.load(f)
            self.all_topic_embeddings = {
                topic_id: np.array(emb) for topic_id, emb in stored_embeddings.items()
            }

    def _prune_facts(self, max_age_days: int = 365, min_importance: int = 7) -> None:
        """drop stale, low-importance facts based on timestamp and importance"""
        if not self.facts:
            return
        
        cutoff = datetime.utcnow().timestamp() - max_age_days * 24 * 3600
        pruned: Dict[str, Dict] = {}
        for field, data in self.facts.items():
            importance = int(data.get("importance", 5))
            ts = data.get("updated_at")
            try:
                updated_ts = datetime.fromisoformat(ts).timestamp() if ts else None
            except Exception:
                updated_ts = None
            
            # keep if high-importance or recent enough
            if importance >= min_importance:
                pruned[field] = data
            elif updated_ts is None or updated_ts >= cutoff:
                pruned[field] = data
        
        self.facts = pruned

    def _maybe_compress_notepad(self, max_chars: int = 8000) -> None:
        """compress notepad with model when it grows very large"""
        if not self.notepad or len(self.notepad) <= max_chars:
            return
        
        prompt = compress_notepad_prompt(self.notepad)
        response = generate_with_retry(prompt, NotepadResponse)
        self.notepad = response.updated_notepad

    def clear(self, *, profile: bool = True, topics: bool = True, notepad: bool = True) -> None:
        """clear selected parts of long-term memory from disk and ram"""
        if profile:
            self.facts = {}
            if self.facts_file.exists():
                self.facts_file.unlink()
        
        if topics:
            self.all_session_topics = []
            self.all_topic_embeddings = {}
            if self.all_session_topics_file.exists():
                self.all_session_topics_file.unlink()
            if self.all_session_topics_embeddings_file.exists():
                self.all_session_topics_embeddings_file.unlink()
        
        if notepad:
            self.notepad = ""
            if self.notepad_file.exists():
                self.notepad_file.unlink()

