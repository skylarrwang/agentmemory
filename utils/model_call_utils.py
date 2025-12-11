from utils.models import OpenAIModels, CloseTopicResponse, FactsResponse
from utils.openai_base import generate_with_retry
from utils.prompts import (
    summarize_conversation_prompt,
    extract_facts_prompt,
    close_topic_prompt,
)

def summarize(text: str, model: str = OpenAIModels.gpt_4o_mini) -> str:
    """Summarize a conversation segment in 2–3 sentences."""
    prompt = summarize_conversation_prompt(text)
    # We reuse the plain text generate() elsewhere; keep this for future structured summaries.
    return generate_with_retry(prompt, CloseTopicResponse, model).summary

def extract_facts(
    user_message: str,
    assistant_response: str,
    model: str = OpenAIModels.gpt_4o_mini,
) -> FactsResponse:
    """Extract concrete user facts plus importance scores."""
    prompt = extract_facts_prompt(user_message, assistant_response)
    return generate_with_retry(prompt, FactsResponse, model)
    

def close_topic(messages_text: str, model: str = OpenAIModels.gpt_4o_mini) -> CloseTopicResponse:
    """Generate topic label and summary from conversation messages
    
    Returns CloseTopicResponse with label and summary fields.
    """
    prompt = close_topic_prompt(messages_text)
    return generate_with_retry(prompt, CloseTopicResponse, model)
