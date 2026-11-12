"""Diagnosis: combine signal-processing evidence with Gemini interpretation."""

from src.diagnosis.engine import DiagnosisResult, analyze_signal
from src.diagnosis.interpreter import (
    answer_question,
    ask_with_signal_tools,
    generate_report,
    interpret_diagnosis,
)

__all__ = [
    "DiagnosisResult",
    "analyze_signal",
    "answer_question",
    "ask_with_signal_tools",
    "interpret_diagnosis",
    "generate_report",
]
