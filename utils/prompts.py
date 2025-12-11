def summarize_conversation_prompt(text: str) -> str:
    """Prompt for summarizing a conversation segment."""
    return f"""Summarize this conversation segment in 2-3 sentences.
Focus on the main topics discussed and key information shared.

Conversation:
{text}"""


def extract_facts_prompt(user_message: str, assistant_response: str) -> str:
    """Prompt for extracting concrete user facts from an exchange."""
    return f"""Extract ONLY genuinely important facts about the user from this exchange.
Be very selective - only extract facts that are:
- Highly stable and important (e.g. name, core values, long-term preferences, significant life details)
- Useful for future conversations and personalization

DO NOT extract:
- Temporary preferences ("I'm craving pizza today")
- Casual mentions without significance
- Inferences or assumptions
- Facts already well-established

For each fact, assign an importance/permanence score from 1 to 10:
- 10 = highly important and very stable (e.g. name, core values, major life facts)
- 8-9 = important and relatively stable (e.g. long-term preferences, significant interests)
- 7 = moderately important but worth noting
- Below 7 = too minor or temporary, DO NOT include

Only include facts with importance >= 7.

Return strict JSON in this format:
{{
  "facts": [
    {{
      "field": "name",
      "value": "Skylar",
      "importance": 10
    }}
  ]
}}
If there are no important facts (importance >= 7), return:
{{"facts": []}}

User: {user_message}
Assistant: {assistant_response}"""


def close_topic_prompt(messages_text: str) -> str:
    """Prompt for generating a topic label and summary from a thread."""
    return f"""Analyze this conversation thread and generate:
1. A concise 2-4 word topic label
2. A summary (2-4 sentences) that includes:
   - What was discussed
   - Communication strategies or patterns that worked well (e.g. how the user preferred information, what approaches resonated)
   - Any notable insights about engaging with this user on similar topics

If there are no notable strategies or patterns, focus on the content discussed.

Return JSON in this format:
{{
    "label": "Travel Planning",
    "summary": "The user discussed planning a trip to Japan, including flight options, hotel recommendations, and itinerary suggestions. User responded well to concrete recommendations with specific examples and preferred concise options over lengthy explanations. Direct, actionable suggestions worked best."
}}

Here is the thread:
{messages_text}"""


def topic_switch_decision_prompt(recent_text: str, current_message: str) -> str:
    """Prompt for deciding whether a new message is a topic switch."""
    return f"""Recent conversation:
{recent_text}

New message:
{current_message}

Decide if this is a new topic or a continuation of the recent conversation.
Be conservative: only say it's a new topic if the subject clearly changes.

Return JSON in this exact format:
{{
  "switch": true or false,
  "topic": "<short topic name if switch is true, otherwise empty string>"
}}"""


def topic_label_prompt(initial_message: str) -> str:
    """Prompt for generating a short topic label from an initial message."""
    return f"""Generate a concise 2-4 word topic label for a conversation that starts with:
"{initial_message}"

Examples:
- "Project Implementation"
- "Weekend Plans"
- "Technical Questions"
- "Course Discussion" """


def rigorous_topic_summary_prompt(messages_text: str) -> str:
    """Prompt for generating a more rigorous topic summary when context overflows."""
    return f"""Generate a comprehensive summary of this conversation thread. 
Include all key points, decisions made, and important context.

Thread:
{messages_text}

Return JSON:
{{
    "label": "Topic Name",
    "summary": "Detailed summary covering all important points..."
}}"""


def compress_notepad_prompt(notepad: str) -> str:
    """Prompt for compressing a long notepad."""
    return f"""You are maintaining a long-term notepad about a user.

The notepad below has grown very long. Rewrite it so that:
- Important and stable information is preserved.
- Recently updated or time-sensitive details are kept.
- Redundant, trivial, or outdated details are merged or removed.
- The overall length is significantly shorter but still useful for future conversations.

Return JSON:
{{
  "updated_notepad": "<compressed notepad text>"
}}

Current notepad:
{notepad}"""

