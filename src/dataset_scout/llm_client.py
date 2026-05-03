"""Shared Azure OpenAI / Entra client plumbing.

Both decompose and the strategy assessor (M2b) talk to AOAI through
litellm with Entra-issued bearer tokens. Centralizing the auth + call
shape here keeps the call sites small and the semantics consistent.

Network-free at import time. Heavy imports (`litellm`, `azure-identity`)
are deferred into the function bodies so the metadata-only path stays
fast and unit tests don't pay for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataset_scout.errors import LLMError

if TYPE_CHECKING:
    from dataset_scout.context import ScoutContext


# Entra token scope for Azure Cognitive Services (covers AOAI).
AOAI_SCOPE = "https://cognitiveservices.azure.com/.default"


def make_token_provider() -> Any:
    """Build a fresh bearer-token provider via `DefaultAzureCredential`.

    The credential chain checks: env-var service-principal,
    workload-identity, managed identity, shared-token-cache, Azure CLI
    (`az login`), and interactive browser. Tokens are cached internally
    so we don't add an extra app-level cache.

    Tests stub this whole function — see decompose / strategy test
    files for the `fake_token_provider` fixture pattern.
    """
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except Exception as exc:
        raise LLMError("azure-identity is required for AOAI Entra auth: " + str(exc)) from exc
    return get_bearer_token_provider(DefaultAzureCredential(), AOAI_SCOPE)


def build_completion_kwargs(
    ctx: ScoutContext,
    *,
    messages: list[dict[str, str]],
    response_format: type | None = None,
    timeout_s: float = 30.0,
    token_provider: Any | None = None,
) -> dict[str, Any]:
    """Construct the litellm.completion(**kwargs) for an AOAI call.

    Routing: `model="azure/<deployment>"` with explicit `api_base`,
    `api_version`, and `azure_ad_token_provider`. Requires `ctx` to be
    AOAI-configured; the caller is responsible for that check.

    `response_format` is the Pydantic class the caller will validate
    the parsed JSON against. We pass `{"type": "json_object"}` to the
    LLM (so Azure OpenAI nudges the model toward valid JSON without
    rejecting the request on strict-schema quirks like `dict[str,
    Literal[...]]`); the caller is expected to post-parse with
    `MyModel.model_validate(json.loads(content))` and retry on
    `ValidationError`.
    """
    if not ctx.aoai_configured:
        raise LLMError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_DEPLOYMENT (and run `az login` for Entra "
            "auth)."
        )
    provider = token_provider if token_provider is not None else make_token_provider()
    kwargs: dict[str, Any] = {
        "model": f"azure/{ctx.aoai_deployment}",
        "api_base": ctx.aoai_endpoint,
        "api_version": ctx.aoai_api_version,
        "azure_ad_token_provider": provider,
        "messages": messages,
        "timeout": timeout_s,
    }
    if response_format is not None:
        # Use json_object mode rather than passing the Pydantic class as
        # response_format=. Azure OpenAI's strict-schema validator
        # rejects schemas that contain dict[str, Literal[...]] (the
        # AssessorResponse case), and we already retry on Pydantic
        # validation failure post-parse.
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def import_litellm() -> Any:
    """Import `litellm` lazily, translating import failures into LLMError.

    Also sets `suppress_debug_info = True` on first import so framework
    decoration around exceptions doesn't leak to user stderr.
    """
    try:
        import litellm
    except Exception as exc:
        raise LLMError(f"litellm not importable: {exc}") from exc
    if hasattr(litellm, "suppress_debug_info"):
        litellm.suppress_debug_info = True
    return litellm


def extract_content(response: Any) -> str:
    """Pull the JSON string from a litellm completion response.

    OpenAI-style: `response.choices[0].message.content`. We avoid
    importing litellm response types so the module stays light and
    test mocks can be plain objects.
    """
    try:
        choice = response.choices[0]
        message = choice.message
        content = message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected LLM response shape: {exc}") from exc
    if not isinstance(content, str):
        raise LLMError("LLM response message.content was not a string")
    return content
