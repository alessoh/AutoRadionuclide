"""Anthropic Claude adapter — only imported when the anthropic SDK is installed."""
from __future__ import annotations
from typing import TYPE_CHECKING
from autoradionuclide.domain.models import ModelRequest, ModelResponse, TokenUsage
from autoradionuclide.providers.base import ModelProvider


class AnthropicProvider(ModelProvider):
    """Real Claude adapter. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-6", ledger=None) -> None:
        super().__init__(ledger)
        try:
            import anthropic as _anthropic
            self._client = _anthropic.Anthropic()
        except ImportError:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install anthropic"
            )
        self._model = model

    def _do_complete(self, request: ModelRequest) -> ModelResponse:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in request.messages
        ]
        kwargs = dict(
            model=self._model,
            max_tokens=request.max_tokens,
            messages=messages,
        )
        if request.system:
            kwargs["system"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        msg = self._client.messages.create(**kwargs)
        content = msg.content[0].text if msg.content else ""
        usage = TokenUsage(
            prompt_tokens=msg.usage.input_tokens,
            completion_tokens=msg.usage.output_tokens,
            total_tokens=msg.usage.input_tokens + msg.usage.output_tokens,
        )
        return ModelResponse(
            request_id=request.request_id,
            model=self._model,
            content=content,
            usage=usage,
        )
