from memory.manager import MemoryManager
from utils.openai_base import generate
from utils.model_call_utils import extract_facts
from concurrent.futures import ThreadPoolExecutor

class Agent:
    """main agent that handles conversations and memory"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = MemoryManager(user_id)
        self.message_count = 0
    
    def single_turn_chat(self, user_message: str) -> str:
        """handle a single turn of chat and update memory"""
        self.message_count += 1
        
        # extract facts in parallel with response gen
        # keep executor alive until after add_single_turn (for topic summarization)
        if user_message.strip():
            executor = ThreadPoolExecutor(max_workers=3)
            try:
                # start fact extraction in parallel
                facts_future = executor.submit(extract_facts, user_message, "")
                
                # get context and generate response (parallel with fact extraction)
                context = self.get_context(user_message)
                response = self.generate_response(context, user_message)
                
                # wait for fact extraction
                facts = facts_future.result()
                
                # save exchange (may close topic, summary gen runs in parallel)
                summary_future = self.memory.add_single_turn(user_message, response, executor)
                
                # wait for topic summary before returning (needed for next query's context)
                if summary_future is not None:
                    closed_topic = summary_future.result()
                    self.memory._persist_closed_topic(closed_topic)
                
                # save facts if we got any
                if facts and facts.facts:
                    self.memory.save_facts(facts)
            finally:
                executor.shutdown(wait=True)
        else:
            # no user message, just generate response
            context = self.get_context(user_message)
            response = self.generate_response(context, user_message)
            facts = None
            
            # save exchange (no executor needed)
            self.memory.add_single_turn(user_message, response, executor=None)
        
        return response
    
    def get_context(self, query: str) -> dict:
        """build context from both short-term and long-term memory"""
        return self.memory.get_context(query)
    
    def generate_response(self, context: dict, query: str) -> str:
        """generate llm response using context"""
        prompt = self.build_prompt(context, query)
        return generate(prompt)
    
    def build_prompt(self, context: dict, query: str) -> str:
        """build prompt from context
        
        assembles in order: user facts, notepad, relevant lt topics, relevant st topics,
        recent conversation, then current query
        """
        parts = []
        
        # user facts/profile
        facts = context['long_term'].get('facts', [])
        if facts:
            parts.append("=== User Profile ===")
            for fact in facts:
                parts.append(f"- {fact}")
            parts.append("")
        
        # notepad (strategic insights)
        notepad = context['long_term'].get('notepad', '')
        if notepad and notepad.strip():
            parts.append("=== Interaction Guidelines (Notepad) ===")
            parts.append(notepad)
            parts.append("")
        
        # relevant long-term topics (previous sessions)
        long_term_topics = context['long_term'].get('relevant_topics', [])
        if long_term_topics:
            parts.append("=== Relevant Past Topics (Previous Sessions) ===")
            for topic in long_term_topics:
                topic_name = topic.get('name', 'Unnamed Topic')
                topic_summary = topic.get('summary', '')
                if topic_summary:
                    parts.append(f"- {topic_name}: {topic_summary}")
                else:
                    parts.append(f"- {topic_name}")
            parts.append("")
        
        # all closed topics from current session (even if not semantically relevant)
        closed_topics_summaries = context['short_term'].get('closed_topics_summaries', [])
        if closed_topics_summaries:
            parts.append("=== Previous Topics (Current Session) ===")
            for topic in closed_topics_summaries:
                topic_name = topic.get('name', 'Unnamed Topic')
                topic_summary = topic.get('summary', '')
                if topic_summary:
                    parts.append(f"- {topic_name}: {topic_summary}")
                else:
                    parts.append(f"- {topic_name}")
            parts.append("")
        
        # relevant short-term topics (includes full threads)
        short_term_topics = context['short_term'].get('relevant_topics', [])
        if short_term_topics:
            parts.append("=== Relevant Topics (Current Session) ===")
            for topic in short_term_topics:
                topic_name = topic.get('name', 'Unnamed Topic')
                topic_summary = topic.get('summary', '')
                topic_messages = topic.get('messages', [])
                
                if topic_summary:
                    parts.append(f"{topic_name}: {topic_summary}")
                else:
                    parts.append(f"{topic_name}:")
                
                # include full message thread for this topic
                if topic_messages:
                    parts.append("  Conversation thread:")
                    for msg in topic_messages:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        parts.append(f"    {role}: {content}")
                parts.append("")
            parts.append("")
        
        # recent conversation (current topic thread)
        recent_messages = context['short_term'].get('recent_messages', [])
        if recent_messages:
            parts.append("=== Recent Conversation ===")
            for msg in recent_messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                parts.append(f"{role}: {content}")
            parts.append("")
        
        # current query
        parts.append(f"User: {query}")
        parts.append("Assistant:")
        
        return "\n".join(parts)
    
    def end_session(self):
        """save everything when session ends
        
        closes any open topic, updates notepad
        """
        self.memory.end_session()
        
