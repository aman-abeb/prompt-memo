"""OpenAI-compatible Pydantic models for chat completion requests and responses."""

from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat completion message schema."""

    role: str = Field(..., description="The role of the message author (system, user, assistant).")
    content: Optional[str] = Field("", description="The contents of the message.")
    name: Optional[str] = Field(None, description="Optional name for the author.")


class ChatCompletionChoice(BaseModel):
    """Individual choice generated in a chat completion."""

    index: int = 0
    message: Message
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    """Token usage information for the completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response payload."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
    cached: bool = Field(False, description="Indicates whether the response was served from cache.")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request payload."""

    model: str = Field(..., description="ID of the model to use.")
    messages: List[Message] = Field(..., description="A list of messages comprising the conversation so far.")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    extra_params: Optional[dict[str, Any]] = None
