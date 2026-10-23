"""OpenAI-compatible Pydantic models for chat completion requests, responses, and metrics."""

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


class ModelPricing(BaseModel):
    """Token pricing rates for an LLM model, expressed in USD per 1,000,000 tokens."""

    prompt_price_per_million: float = Field(..., description="USD cost per 1M prompt/input tokens.")
    completion_price_per_million: float = Field(..., description="USD cost per 1M completion/output tokens.")

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculates total dollar cost for the given token counts."""
        prompt_cost = (prompt_tokens / 1_000_000.0) * self.prompt_price_per_million
        completion_cost = (completion_tokens / 1_000_000.0) * self.completion_price_per_million
        return prompt_cost + completion_cost


DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(prompt_price_per_million=2.50, completion_price_per_million=10.00),
    "gpt-4o-mini": ModelPricing(prompt_price_per_million=0.15, completion_price_per_million=0.60),
    "gpt-4-turbo": ModelPricing(prompt_price_per_million=10.00, completion_price_per_million=30.00),
    "gpt-4": ModelPricing(prompt_price_per_million=30.00, completion_price_per_million=60.00),
    "gpt-3.5-turbo": ModelPricing(prompt_price_per_million=0.50, completion_price_per_million=1.50),
    "claude-3-5-sonnet": ModelPricing(prompt_price_per_million=3.00, completion_price_per_million=15.00),
    "claude-3-opus": ModelPricing(prompt_price_per_million=15.00, completion_price_per_million=75.00),
    "claude-3-haiku": ModelPricing(prompt_price_per_million=0.25, completion_price_per_million=1.25),
}

FALLBACK_MODEL_PRICING = ModelPricing(prompt_price_per_million=2.50, completion_price_per_million=10.00)


def get_model_pricing(
    model: str,
    custom_pricing: Optional[dict[str, ModelPricing]] = None,
) -> ModelPricing:
    """Finds pricing model for a given model identifier with prefix and fallback resolution."""
    pricing_map = custom_pricing if custom_pricing is not None else DEFAULT_MODEL_PRICING
    clean_model = model.strip().lower()

    if clean_model in pricing_map:
        return pricing_map[clean_model]

    for key, val in pricing_map.items():
        if clean_model.startswith(key.lower()):
            return val

    for key, val in pricing_map.items():
        if key.lower() in clean_model:
            return val

    return FALLBACK_MODEL_PRICING


class ModelMetrics(BaseModel):
    """Aggregated cache and savings metrics for a specific model."""

    entries: int = Field(0, description="Active cache entries for this model.")
    total_hits: int = Field(0, description="Total cache hits for this model.")
    exact_hits: int = Field(0, description="Exact cache hits for this model.")
    fuzzy_hits: int = Field(0, description="Fuzzy MinHash cache hits for this model.")
    total_misses: int = Field(0, description="Total cache misses recorded for this model.")
    saved_prompt_tokens: int = Field(0, description="Cumulative prompt tokens saved.")
    saved_completion_tokens: int = Field(0, description="Cumulative completion tokens saved.")
    saved_total_tokens: int = Field(0, description="Cumulative total tokens saved.")
    estimated_dollar_savings: float = Field(0.0, description="Estimated dollar savings based on model pricing.")


class MetricsResponse(BaseModel):
    """System-wide cache performance, token savings, and cost analytics response schema."""

    total_entries: int = Field(0, description="Total active items in the cache.")
    total_requests: int = Field(0, description="Total completion requests processed (hits + misses).")
    total_hits: int = Field(0, description="Total cache hits (exact + fuzzy).")
    exact_hits: int = Field(0, description="Total exact cache hits.")
    fuzzy_hits: int = Field(0, description="Total fuzzy MinHash cache hits.")
    total_misses: int = Field(0, description="Total cache misses.")
    hit_rate: float = Field(0.0, description="Cache hit rate (hits / requests) between 0.0 and 1.0.")
    saved_prompt_tokens: int = Field(0, description="Total prompt tokens saved across all hits.")
    saved_completion_tokens: int = Field(0, description="Total completion tokens saved across all hits.")
    saved_total_tokens: int = Field(0, description="Total tokens saved across all hits.")
    estimated_dollar_savings: float = Field(0.0, description="Total estimated dollar savings.")
    by_model: dict[str, ModelMetrics] = Field(default_factory=dict, description="Breakdown by model.")
    pricing: dict[str, ModelPricing] = Field(default_factory=dict, description="Active pricing table used.")
