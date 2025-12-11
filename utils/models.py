from pydantic import BaseModel, Field
from typing import List


class CloseTopicResponse(BaseModel):
    """Model for closing a topic."""
    label: str = Field(description="2-4 word topic label")
    summary: str = Field(description="2-3 sentence summary of the topic")


class TopicSwitchDecision(BaseModel):
    """Structured response for topic switch detection."""
    switch: bool = Field(description="True if this is a new topic, otherwise False")
    topic: str = Field(
        default="",
        description="Short topic name if switch is True, otherwise empty string",
    )


class NotepadResponse(BaseModel):
    """Model for updating the notepad."""
    updated_notepad: str = Field(description="Updated notepad content")


class Fact(BaseModel):
    """Single extracted user fact."""
    field: str = Field(description="Name of the fact field, e.g. 'name', 'location'")
    value: str = Field(description="Concrete fact value stated by the user")
    importance: int = Field(
        description="Importance/permanence from 1 (low, temporary) to 10 (high, stable)",
        ge=1,
        le=10,
    )


class FactsResponse(BaseModel):
    """Structured collection of extracted facts."""
    facts: List[Fact] = Field(
        default_factory=list,
        description="List of concrete user facts extracted from the exchange",
    )


class OpenAIModels:
    """Model name constants."""
    gpt_4o_mini: str = "gpt-4o-mini"
    gpt_4o: str = "gpt-4o"

