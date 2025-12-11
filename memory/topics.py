from typing import Optional, List, Dict

class Topic:
    """A single conversation topic.
    Note that the live / in-memory version is more verbose than storing in json, 
    this is bc less info is needed when retrieving over json
    """
    def __init__(self, id: str, name: str, summary: Optional[str], messages: List[Dict], created_at: str, closed_at: Optional[str]):
        self.id = id
        self.name = name
        self.summary = summary
        self.messages = messages
        self.num_messages = 0
        self.default_k = 20
        self.created_at = created_at
        self.closed_at = closed_at
        self.embedding: List[float] = None
    
    def add_message_to_topic(self, message: Dict):
        self.messages.append(message)
        self.num_messages += 1
    
    def get_default_context(self, recent_k: int = 5):
        """Returns summary + most recent k messages in a formatted string"""
        return f"""
        Summary: {self.summary}\n
        Most recent {recent_k} messages: {self.get_most_recent_k_messages(recent_k)}\n
        """
    
    def get_most_recent_k_messages(self, recent_k: Optional[int] = None):
        """Get the most recent k messages from this topic"""
        if recent_k is None:
            recent_k = self.default_k
        if self.num_messages < recent_k:
            return self.messages
        else:
            i = self.num_messages - recent_k
            return self.messages[i:]


    


