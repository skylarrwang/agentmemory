from pydantic import BaseModel, ValidationError
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import time

from utils.models import OpenAIModels

load_dotenv()
client = OpenAI()


def _extract_text(response) -> str:
    """Extract plain text from a Responses API response."""
    try:
        outputs = getattr(response, "output", None) or []
        texts: list[str] = []
        for msg in outputs:
            for c in getattr(msg, "content", []) or []:
                text = getattr(c, "text", None)
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass
    # Fallback to string repr if structure changes
    return str(response)


def generate(prompt: str, model: str = OpenAIModels.gpt_4o_mini) -> str:
    """Generate a text response."""
    response = client.responses.create(
        model=model,
        input=f"You are a helpful assistant.\n\n{prompt}",
    )
    return _extract_text(response)

def generate_with_retry(
    prompt: str,
    output_class: type[BaseModel],
    model: str = OpenAIModels.gpt_4o_mini,
    max_retries: int = 3,
) -> BaseModel:
    """Generate response and parse as structured output with retry logic.

    Tries to parse JSON from the model's text output and validate against a Pydantic model.
    Retries with exponential backoff if parsing/validation fails.
    """
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
            )

            # Extract just the assistant text, not the whole Response repr.
            response_text = _extract_text(response)

            # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
            json_text = response_text.strip()
            if json_text.startswith("```"):
                # Remove opening fence (```json or ```)
                json_text = re.sub(r"^```(?:json)?\s*\n?", "", json_text, flags=re.MULTILINE)
                # Remove closing fence (```)
                json_text = re.sub(r"\n?```\s*$", "", json_text, flags=re.MULTILINE)
                json_text = json_text.strip()

            # Try to extract JSON object if wrapped in other text
            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()

            # Parse and validate
            data = json.loads(json_text)
            validated = output_class(**data)
            return validated

        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (attempt + 1)  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                raise ValueError(f"Could not parse response as {output_class.__name__}: {e}")
        except Exception:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (attempt + 1)
                time.sleep(wait_time)
                continue
            else:
                raise
