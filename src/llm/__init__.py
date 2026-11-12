"""Gemini LLM integration and Function Calling tools (official ``google-genai`` SDK)."""

from src.llm.gemini_client import GeminiClient, GeminiError, GeminiResponse
from src.llm.tools import (
    ALLOWED_TOOLS,
    TOOL_DECLARATIONS,
    ToolCallRecord,
    ToolExecutionContext,
    analyze_vibration_signal,
    calculate_bearing_frequencies,
    calculate_psd,
    calculate_signal_features,
    compare_fault_frequencies,
    execute_tool_call,
    perform_fft_analysis,
    search_bearing_knowledge,
)

__all__ = [
    "ALLOWED_TOOLS",
    "GeminiClient",
    "GeminiError",
    "GeminiResponse",
    "TOOL_DECLARATIONS",
    "ToolCallRecord",
    "ToolExecutionContext",
    "analyze_vibration_signal",
    "calculate_bearing_frequencies",
    "calculate_psd",
    "calculate_signal_features",
    "compare_fault_frequencies",
    "execute_tool_call",
    "perform_fft_analysis",
    "search_bearing_knowledge",
]
