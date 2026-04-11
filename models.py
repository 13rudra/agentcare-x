"""
AgentCare X — Pydantic Data Models

All typed models for the OpenEnv customer operations environment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Order & Tool Schemas
# ---------------------------------------------------------------------------

class OrderInfo(BaseModel):
    """Structured representation of a customer order."""
    order_id: str
    product: str
    # FIXED — extended with new task statuses
    status: Literal["shipped", "delivered", "wrong_item", "delayed", "out_of_stock", "active_subscription"]
    amount: float
    order_date: str
    estimated_delivery: str


class ToolSpec(BaseModel):
    """Describes one tool available to the agent."""
    name: str
    description: str
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of parameter name → type/description string",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Everything the agent sees at each step."""
    customer_message: str
    emotion_level: float = Field(ge=0.0, le=1.0)
    order_info: OrderInfo
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description='List of {"role": "customer"|"agent", "content": "..."}',
    )
    available_tools: list[ToolSpec] = Field(default_factory=list)
    last_action_feedback: str | None = None
    instructions: str = ""
    turn_number: int = 0
    max_turns: int = 10


# ---------------------------------------------------------------------------
# Action (agent output)
# ---------------------------------------------------------------------------

class AgentAction(BaseModel):
    """
    Structured JSON action emitted by the agent.

    Two patterns:
      1. respond  → message is required
      2. call_tool → tool_name and tool_parameters are required
    """
    action_type: Literal["respond", "call_tool"]
    message: str | None = None
    tool_name: str | None = None
    tool_parameters: dict | None = None

    @model_validator(mode='after')
    def check_fields(self) -> 'AgentAction':
        if self.action_type == 'respond' and not self.message:
            raise ValueError("action_type='respond' requires 'message' to be provided.")
        if self.action_type == 'call_tool' and not self.tool_name:
            raise ValueError("action_type='call_tool' requires 'tool_name' to be provided.")
        if self.action_type == 'call_tool' and self.tool_parameters is None:
            self.tool_parameters = {}
        return self


# ---------------------------------------------------------------------------
# Environment State
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Full internal snapshot returned by state()."""
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 10
    resolved: bool = False
    failed: bool = False
    failure_reason: str | None = None
    emotion_history: list[float] = Field(default_factory=list)
    tool_calls_made: list[dict] = Field(default_factory=list)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    # NEW — track hallucinated tool calls
    hallucination_count: int = 0


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardInfo(BaseModel):
    """Per-step reward breakdown."""
    total: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)
