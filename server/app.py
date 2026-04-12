"""
AgentCare X — FastAPI Server
OpenEnv-compliant REST API for the customer support RL environment.
"""

from __future__ import annotations

import sys
import pathlib

# Ensure project root is always in Python path (required for Docker)
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models import AgentAction, EnvState
from env.environment import AgentCareEnv
from tasks.task_easy import TASK as TASK_EASY
from tasks.task_medium import TASK as TASK_MEDIUM
from tasks.task_hard import TASK as TASK_HARD
from tasks.task_out_of_stock import TASK as TASK_OUT_OF_STOCK
from tasks.task_subscription import TASK as TASK_SUBSCRIPTION

app = FastAPI(
    title="AgentCare X",
    description="AI Customer Operations Environment — OpenEnv compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_DASHBOARD_DIR = _ROOT / "dashboard"
env = AgentCareEnv()


class TaskInfo(BaseModel):
    task_id: str
    difficulty: str
    description: str
    expected_steps: int
    required_tools: list[str]


@app.post("/reset")
async def reset_env(request: Request):
    """Reset environment — required by OpenEnv automated checker."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id = body.get("task_id", "easy")
    try:
        obs = env.reset(task_id=task_id)
        return {
            "observation": obs.model_dump(),
            "reward": 0.01,
            "done": False,
            "info": {}
        }
    except Exception as e:
        return {
            "observation": {"customer_message": "Error loading task", "emotion_level": 0.5},
            "reward": 0.01,
            "done": True,
            "info": {"error": str(e)}
        }


@app.post("/step")
async def step_env(request: Request):
    """Execute one agent action."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    action_type = body.get("action_type") or "respond"
    message = body.get("message")

    if action_type == "respond" and not message:
        message = "I am processing your request now."

    tool_name = body.get("tool_name")
    if action_type == "call_tool" and not tool_name:
        action_type = "respond"
        message = "I encountered an error trying to use a tool."

    tool_parameters = body.get("tool_parameters", {})

    try:
        action = AgentAction(
            action_type=action_type,
            message=message,
            tool_name=tool_name,
            tool_parameters=tool_parameters
        )
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": round(max(0.01, min(0.99, float(reward))), 4),
            "done": bool(done),
            "info": info
        }
    except Exception as e:
        return {
            "observation": {"customer_message": "Crash prevented.", "emotion_level": 0.5},
            "reward": 0.01,
            "done": True,
            "info": {"error": str(e)}
        }


@app.get("/state")
def get_state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    tasks = [TASK_EASY, TASK_MEDIUM, TASK_HARD, TASK_OUT_OF_STOCK, TASK_SUBSCRIPTION]
    return [
        TaskInfo(
            task_id=t["task_id"],
            difficulty=t["difficulty"],
            description=t["description"],
            expected_steps=t["expected_steps"],
            required_tools=t["required_tools"],
        )
        for t in tasks
    ]


@app.get("/health")
def health():
    """Health check — required by OpenEnv automated checker."""
    return {"status": "ok", "name": "agentcare-x", "version": "1.0.0"}


@app.get("/")
def root():
    if _DASHBOARD_DIR.exists():
        return RedirectResponse(url="/dashboard/index.html")
    return {"name": "AgentCare X", "version": "1.0.0", "status": "running"}


# Mount dashboard LAST so it doesn't override API routes
if _DASHBOARD_DIR.exists():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_DASHBOARD_DIR), html=True),
        name="dashboard"
    )


def main():
    """Entry point for [project.scripts] — required by openenv validate."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
