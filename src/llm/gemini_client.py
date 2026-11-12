"""Thin, testable wrapper around the official Google Gen AI SDK.

Uses the Interactions API (``client.interactions.create``) for text generation
and multi-step Function Calling, and ``client.models.embed_content`` for
embeddings used by the RAG module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from google import genai
from google.genai import errors, types

from src.config import Settings, get_settings
from src.llm.tools import (
    TOOL_DECLARATIONS,
    ToolCallRecord,
    ToolExecutionContext,
    execute_tool_call,
)
from src.logger import get_logger

logger = get_logger("llm.gemini")

FALLBACK_MODEL = "gemini-3.5-flash-lite"


class GeminiError(RuntimeError):
    """User-friendly error raised for any Gemini API failure."""


@dataclass
class GeminiResponse:
    text: str
    interaction_id: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


def _friendly_api_error(exc: Exception) -> str:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    hints = {
        400: "Bad request - check the model name and request parameters.",
        401: "Authentication failed - check GEMINI_API_KEY.",
        403: "Permission denied - the API key is invalid or lacks access.",
        404: "Model or resource not found - check GEMINI_MODEL.",
        429: "Rate limit / quota exceeded - wait and retry.",
        500: "Gemini server error - retry shortly.",
        503: "Gemini service unavailable (high demand) - retry shortly or switch GEMINI_MODEL.",
    }
    hint = hints.get(code, "Unexpected Gemini API error.")
    return f"[{code}] {hint} ({getattr(exc, 'message', '') or exc})"


class GeminiClient:
    """High-level Gemini client used across VibraMind.

    The API key is read from settings (``.env``) and passed straight to the SDK;
    it is never stored on this object or logged.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        api_key = self.settings.require_api_key()
        self.model = self.settings.gemini_model
        self.embedding_model = self.settings.gemini_embedding_model
        try:
            self._client = genai.Client(api_key=api_key)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to initialise Gemini client")
            raise GeminiError(f"Could not initialise Gemini client: {exc}") from exc
        logger.info("Gemini client initialised (model=%s)", self.model)

    def _create_interaction(self, **kwargs: Any) -> Any:
        """Call ``client.interactions.create`` with automatic 429/503 fallback."""
        try:
            return self._client.interactions.create(**kwargs)
        except Exception as exc:
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if code in (429, 503) and kwargs.get("model") != FALLBACK_MODEL:
                logger.warning(
                    "Model %s returned %s; retrying with fallback model %s",
                    kwargs.get("model"),
                    code,
                    FALLBACK_MODEL,
                )
                try:
                    fallback_kwargs = {**kwargs, "model": FALLBACK_MODEL}
                    return self._client.interactions.create(**fallback_kwargs)
                except Exception as fallback_exc:
                    msg = _friendly_api_error(fallback_exc)
                    logger.error("Gemini fallback interaction failed: %s", msg)
                    raise GeminiError(msg) from fallback_exc
            if isinstance(exc, errors.APIError) or code is not None:
                msg = _friendly_api_error(exc)
                logger.error("Gemini interaction failed: %s", msg)
                raise GeminiError(msg) from exc
            logger.exception("Unexpected error calling Gemini")
            raise GeminiError(f"Unexpected error calling Gemini: {exc}") from exc

    # ------------------------------------------------------------------ text
    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        previous_interaction_id: str | None = None,
        max_output_tokens: int | None = None,
    ) -> GeminiResponse:
        """Generate text for ``prompt``."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        kwargs: dict[str, Any] = {"model": self.model, "input": prompt}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if previous_interaction_id:
            kwargs["previous_interaction_id"] = previous_interaction_id
        if max_output_tokens:
            kwargs["generation_config"] = {"max_output_tokens": max_output_tokens}

        interaction = self._create_interaction(**kwargs)
        text = interaction.output_text or ""
        if not text:
            logger.warning("Gemini returned an empty response (status=%s)", interaction.status)
        return GeminiResponse(text=text, interaction_id=interaction.id)

    # ----------------------------------------------------- function calling
    def generate_with_tools(
        self,
        prompt: str,
        tool_context: ToolExecutionContext | None = None,
        *,
        system_instruction: str | None = None,
        previous_interaction_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tool_rounds: int = 5,
    ) -> GeminiResponse:
        """Run a multi-step Gemini Function Calling interaction loop.

        1. Sends ``prompt`` with ``tools`` (defaults to ``TOOL_DECLARATIONS``).
        2. Whenever Gemini emits ``function_call`` steps, executes the whitelisted
           Python functions against ``tool_context`` and sends their JSON numerical
           outputs back as ``function_result`` steps via ``previous_interaction_id``.
        3. Returns the final model interpretation along with all executed ``tool_calls``.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        active_tools = tools if tools is not None else TOOL_DECLARATIONS
        recorded_calls: list[ToolCallRecord] = []

        req_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "tools": active_tools,
        }
        if system_instruction:
            req_kwargs["system_instruction"] = system_instruction
        if previous_interaction_id:
            req_kwargs["previous_interaction_id"] = previous_interaction_id

        interaction = self._create_interaction(**req_kwargs)

        for _round in range(max_tool_rounds):
            steps = interaction.steps or []
            fc_steps = [s for s in steps if getattr(s, "type", None) == "function_call"]
            if not fc_steps:
                break

            result_inputs: list[dict[str, Any]] = []
            for fc in fc_steps:
                fn_name = str(getattr(fc, "name", ""))
                fn_args = dict(getattr(fc, "arguments", {}) or {})
                call_id = str(getattr(fc, "id", ""))

                output_dict = execute_tool_call(fn_name, fn_args, context=tool_context)
                recorded_calls.append(
                    ToolCallRecord(name=fn_name, arguments=fn_args, output=output_dict)
                )
                result_inputs.append(
                    {
                        "type": "function_result",
                        "call_id": call_id,
                        "name": fn_name,
                        "result": json.dumps(output_dict),
                    }
                )

            follow_kwargs: dict[str, Any] = {
                "model": self.model,
                "previous_interaction_id": interaction.id,
                "input": result_inputs,
                "tools": active_tools,
            }
            if system_instruction:
                follow_kwargs["system_instruction"] = system_instruction

            interaction = self._create_interaction(**follow_kwargs)

        text = interaction.output_text or ""
        return GeminiResponse(
            text=text,
            interaction_id=interaction.id,
            tool_calls=recorded_calls,
        )

    # ------------------------------------------------------------ embeddings
    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """Embed a list of texts. ``task_type`` is RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = 50
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = self._client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                vectors.extend(list(e.values) for e in response.embeddings)
        except errors.APIError as exc:
            msg = _friendly_api_error(exc)
            logger.error("Gemini embedding failed: %s", msg)
            raise GeminiError(msg) from exc
        except Exception as exc:
            logger.exception("Unexpected error creating embeddings")
            raise GeminiError(f"Unexpected error creating embeddings: {exc}") from exc
        return vectors

    # -------------------------------------------------------------- health
    def test_connection(self) -> tuple[bool, str]:
        """Send a tiny request to verify the key, model and network work."""
        try:
            response = self.generate(
                "Reply with exactly: VibraMind connection OK",
                max_output_tokens=256,
            )
        except GeminiError as exc:
            return False, str(exc)
        reply = response.text.strip()
        if not reply:
            return False, "Connected, but Gemini returned an empty response."
        logger.info("Gemini connection test succeeded")
        return True, reply
