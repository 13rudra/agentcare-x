"""
AgentCare X — Dense Reward Calcswegsgesgulator

Computes per-step reward signals based on agent actions and environment state.
All signals are deterministic and rule-based.
"""

from __future__ import annotations

from models import AgentAction, RewardInfo
from env.customer import detect_empathy, detect_rudeness
from env.empathy_judge import judge_empathy  # NEW


def compute_reward(
    action: AgentAction,
    *,
    tool_result: dict | None,
    required_tools: list[str],
    tools_already_called: list[str],
    previous_actions: list[dict],
    step_count: int,
    expected_steps: int,
    max_steps: int,
    resolved_this_step: bool,
    emotion_delta: float,
    customer_message_pending: bool,
    # NEW — hallucination flag from tool execution
    hallucinated_tool: bool = False,
    # NEW — use LLM judge for empathy (disabled in mock mode)
    use_llm_judge: bool = False,
) -> RewardInfo:
    """
    Compute dense per-step reward.

    Returns RewardInfo with named breakdown of all signals.
    """
    breakdown: dict[str, float] = {}

    # --- NEW — Hallucinated tool penalty ---
    if hallucinated_tool:
        breakdown["hallucination_penalty"] = -0.20

    # --- Tool usage signals ---
    if action.action_type == "call_tool" and action.tool_name:
        if not hallucinated_tool:  # Only score real tools
            if action.tool_name in required_tools:
                if action.tool_name not in tools_already_called:
                    breakdown["correct_tool"] = 0.20
                else:
                    breakdown["redundant_tool"] = -0.10
            else:
                breakdown["wrong_tool"] = -0.15

            # Check if tool succeeded
            if tool_result and tool_result.get("success"):
                breakdown["progress"] = 0.15
            elif tool_result and tool_result.get("error"):
                breakdown["tool_error"] = -0.05

    # --- Response quality signals ---
    if action.action_type == "respond" and action.message:
        # NEW — LLM-as-a-Judge empathy scoring (replaces keyword boolean)
        if use_llm_judge:
            empathy_score = judge_empathy(action.message)
            # Scale: 0.0–1.0 → 0.0–0.15 bonus
            empathy_reward = round(empathy_score * 0.15, 4)
            if empathy_reward > 0:
                breakdown["empathy_bonus"] = empathy_reward
        else:
            # Fallback: keyword-based (original behavior)
            if detect_empathy(action.message):
                breakdown["empathy_bonus"] = 0.10

        if detect_rudeness(action.message):
            breakdown["rude_penalty"] = -0.20

    # --- Resolution ---
    if resolved_this_step:
        breakdown["resolution"] = 0.30

    # --- Redundant action (exact duplicate of previous) ---
    current_action_dict = action.model_dump()
    repetition_count = sum(1 for prev in previous_actions if prev == current_action_dict)
    if repetition_count > 0:
        breakdown["redundant_step"] = -0.10 * repetition_count

    # --- Ignored customer (calling tool without acknowledging new message) ---
    if action.action_type == "call_tool" and customer_message_pending:
        # Check if the agent has responded at all in recent history
        last_was_tool = len(previous_actions) > 0 and previous_actions[-1].get("action_type") == "call_tool"
        if last_was_tool:
            breakdown["ignored_customer"] = -0.15

    # --- Emotion tracking ---
    if emotion_delta > 0:
        breakdown["frustration_spike"] = -0.05

    # --- Efficiency bonus (only at episode end) ---
    if resolved_this_step and step_count <= expected_steps:
        breakdown["efficiency_bonus"] = 0.10

    total = sum(breakdown.values())
    # Clamp to (0.01, 0.99) — validator requires scores never exactly 0 or 1
    total = max(0.01, min(0.99, total))


    return RewardInfo(total=round(total, 4), breakdown=breakdown)
