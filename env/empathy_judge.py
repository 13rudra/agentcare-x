"""
AgentCare X — LLM-as-a-Judge Empathy Scorer

Replaces keyword-based empathy detection with a lightweight LLM call
that scores agent responses for empathy and helpfulness on 0.0–1.0.

Falls back to keyword-based scoring if the LLM call fails.
"""

from __future__ import annotations

import json
import os
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are an expert evaluator of customer service quality.

Rate the following customer service agent response for EMPATHY and HELPFULNESS.

Consider:
- Does the agent acknowledge the customer's feelings?
- Is the tone warm, professional, and caring?
- Does the agent take ownership and offer concrete help?
- Is the response dismissive, robotic, or generic?

Agent response:
\"{agent_message}\"

Return ONLY a JSON object with no additional text:
{{"score": <float 0.0 to 1.0>, "reason": "<brief explanation>"}}

Where 0.0 = completely cold/unhelpful and 1.0 = exceptionally empathetic and helpful."""


# ---------------------------------------------------------------------------
# Judge implementation
# ---------------------------------------------------------------------------

def judge_empathy(agent_message: str) -> float:
    """
    Score an agent message for empathy using a lightweight LLM call.

    Returns a float in [0.0, 1.0].
    Falls back to keyword-based scoring on any failure.
    """
    try:
        return _call_llm_judge(agent_message)
    except Exception as e:
        logger.warning("LLM empathy judge failed (%s), falling back to keywords", e)
        return _keyword_fallback(agent_message)


def _call_llm_judge(agent_message: str) -> float:
    """Call the judge LLM and parse the score."""
    from openai import OpenAI

    judge_api_key = os.environ.get("JUDGE_API_KEY", os.environ.get("HF_TOKEN", ""))
    judge_base_url = os.environ.get("JUDGE_API_BASE_URL", os.environ.get("API_BASE_URL", "https://api.openai.com/v1"))
    judge_model = os.environ.get("JUDGE_MODEL_NAME", "gpt-4o-mini")

    if not judge_api_key or judge_api_key == "your_judge_api_key_here":
        raise ValueError("No judge API key configured")

    client = OpenAI(base_url=judge_base_url, api_key=judge_api_key)
    prompt = _JUDGE_PROMPT.format(agent_message=agent_message)

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
    )

    raw = response.choices[0].message.content or ""

    # Parse JSON from response
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    data = json.loads(raw)
    score = float(data.get("score", 0.5))
    return max(0.01, min(0.99, score))


# ---------------------------------------------------------------------------
# Keyword fallback (graceful degradation)
# ---------------------------------------------------------------------------

_EMPATHY_KEYWORDS: set[str] = {
    "sorry", "apologize", "apologies", "understand", "understand your frustration",
    "help", "assist", "right away", "certainly", "absolutely",
    "inconvenience", "bear with", "patience", "appreciate",
    "let me", "i will", "i'll", "resolve", "fix", "take care",
}


def _keyword_fallback(message: str) -> float:
    """Score empathy via keyword presence — returns 0.0, 0.5, or 0.8."""
    lower = message.lower()
    matches = sum(1 for kw in _EMPATHY_KEYWORDS if kw in lower)
    if matches >= 3:
        return 0.8
    if matches >= 1:
        return 0.5
    return 0.01
