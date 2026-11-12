"""Basic Gemini connection test.

Run either way from the project root:
    python tests/test_gemini_connection.py      # standalone, prints result
    pytest tests/test_gemini_connection.py -v   # skipped if no API key
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.llm import GeminiClient  # noqa: E402

requires_key = pytest.mark.skipif(not get_settings().has_api_key, reason="GEMINI_API_KEY not configured")


@requires_key
def test_gemini_text_generation():
    ok, message = GeminiClient().test_connection()
    assert ok, message


@requires_key
def test_gemini_embeddings():
    vectors = GeminiClient().embed(["bearing outer race fault"], task_type="RETRIEVAL_QUERY")
    assert len(vectors) == 1 and len(vectors[0]) > 0


def main() -> int:
    settings = get_settings()
    if not settings.has_api_key:
        print("❌ GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1
    print(f"Testing Gemini connection (model: {settings.gemini_model})...")
    try:
        client = GeminiClient(settings)
        ok, message = client.test_connection()
    except Exception as exc:
        print(f"❌ Connection failed: {exc}")
        return 1
    print(("✅ Connected. Gemini replied: " if ok else "❌ Connection failed: ") + message)
    if ok:
        try:
            dim = len(client.embed(["test"])[0])
            print(f"✅ Embeddings OK ({settings.gemini_embedding_model}, dim={dim})")
        except Exception as exc:
            print(f"⚠️  Embeddings failed: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

