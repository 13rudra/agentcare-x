<![CDATA[<div align="center">

# 🏥 AgentCare X

**AI Customer Operations Environment**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blueviolet)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](#-docker)

An **OpenEnv-compliant benchmark** for evaluating AI agents on **multi-step reasoning**, **tool orchestration**, and **emotional intelligence** in real-world customer support scenarios.

[Quick Start](#-quick-start) •
[Architecture](#-architecture) •
[Tasks](#-tasks) •
[Grading](#-grading) •
[API Reference](#-api-reference) •
[Contributing](#-contributing)

---

## 🏆 OVERVIEW FOR HUMAN JUDGES

**AgentCare X** is purpose-built to solve a critical gap in LLM evaluation: assessing how models perform in **dynamic, multi-turn customer operations**. 

While standard benchmarks like MMLU test static knowledge, this project evaluates:
1. **Multi-Step Reasoning:** Agents must logically chain multiple actions together without hallucinating paths.
2. **Dynamic Tool Usage:** Agents are evaluated on correctly mapping dynamic backend structures and parameters (e.g., retrieving an order, then conditionally passing its data into a refund tool).
3. **Emotional Intelligence Simulation:** A mathematically modeled customer state tracks frustration. If agents act abruptly or use tools without empathetic communication, the customer's fury will escalate—leading to episode failure.

This directly mirrors **real-world AI evaluation** for scale-ups and enterprises aiming to automate their Level 1 Support tier.

---

## 🎯 Problem Statement

Companies deploying AI agents in customer operations need rigorous evaluation beyond simple accuracy metrics. Real customer interactions demand:

| Capability | Why It Matters |
|---|---|
| **Multi-step reasoning** | Issues often need chained actions: lookup → refund → escalate |
| **Tool orchestration** | Agents must call the right tools with correct parameters |
| **Emotional intelligence** | Gauging and de-escalating customer frustration in real-time |
| **Efficiency** | Resolving issues in minimal steps saves cost and builds trust |

**AgentCare X** provides a **deterministic, reproducible environment** that evaluates all four dimensions simultaneously with dense reward signals and weighted grading.

---

## 🌍 Real-World Impact

This benchmark mirrors **production customer support systems** used by Amazon, Flipkart, and similar platforms. Use it to:

- 🔬 **Evaluate** LLM agents before deployment  
- ⚖️ **Compare** model capabilities on realistic tasks  
- 🤖 **Train** agents via reinforcement learning with dense reward signals  
- 💡 **Benchmark** emotional intelligence in AI systems  

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| pip | Latest |

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/your-org/agentcare-x.git
cd agentcare-x

# Create a virtual environment (recommended)
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference (Mock Mode — No LLM Required)

The mock mode uses **optimal scripted actions** — perfect for testing and validation:

```bash
python inference.py --mock
```

**Expected output:**

```
============================================================
AgentCare X — Inference Run
Mode: MOCK
============================================================

[START] task_id=easy difficulty=easy
[STEP] step=1 action={"action_type": "respond", ...} reward=0.1000 emotion=0.10
[STEP] step=2 action={"action_type": "call_tool", ...} reward=0.3500 emotion=-0.10
[STEP] step=3 action={"action_type": "respond", ...} reward=0.4000 emotion=-0.20
[END] task_id=easy grader_score=0.9500 total_reward=0.8500 steps=3

...

============================================================
RESULTS SUMMARY
============================================================
Task       Diff     Score    Reward     Steps  Status
------------------------------------------------------------
easy       easy     0.9500   0.8500     3      RESOLVED
medium     medium   0.9000   0.8000     5      RESOLVED
hard       hard     0.8500   0.9500     7      RESOLVED
------------------------------------------------------------
Average Score: 0.9000
============================================================
```

### 3. Run Inference (With LLM)

Connect to any **OpenAI-compatible API** (Hugging Face, OpenAI, vLLM, Ollama, etc.):

```bash
# Set environment variables
set API_BASE_URL=https://api-inference.huggingface.co/v1       # Windows
set MODEL_NAME=meta-llama/Llama-3-8b-instruct                   # Windows
set HF_TOKEN=hf_your_token_here                                  # Windows

# Or on macOS/Linux:
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3-8b-instruct"
export HF_TOKEN="hf_your_token_here"

# Run
python inference.py
```

### 4. Run the FastAPI Server

Spin up the **OpenEnv REST API** for external agents:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 🏗️ Architecture

```
agentcare-x/
├── models.py                # Pydantic data models (Observation, Action, State, Reward)
├── __init__.py              # Package exports
├── env/
│   ├── __init__.py
│   ├── environment.py       # Core env: step(), reset(), state()
│   ├── tools.py             # Tool registry (check_order, refund, escalate)
│   ├── customer.py          # Deterministic customer simulator + emotion model
│   └── rewards.py           # Dense per-step reward calculator
├── tasks/
│   ├── __init__.py
│   ├── task_easy.py         # Task 1: Order lookup (calm customer)
│   ├── task_medium.py       # Task 2: Angry customer + refund
│   └── task_hard.py         # Task 3: Wrong item + refund + escalation
├── graders/
│   ├── __init__.py
│   └── grader.py            # Deterministic end-of-episode grader
├── server/
│   ├── __init__.py
│   └── app.py               # FastAPI server (OpenEnv REST API)
├── inference.py             # Baseline LLM agent + mock agent script
├── openenv.yaml             # OpenEnv manifest
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata
├── Dockerfile               # Container definition
├── .dockerignore            # Docker ignore rules
└── README.md                # This file
```

### Data Flow

```
┌─────────────┐     JSON Action      ┌──────────────────┐
│  LLM Agent  │ ──────────────────▶  │   AgentCareEnv   │
│  (or Mock)  │                      │                  │
│             │  ◀──────────────────  │  • step()        │
│             │    Observation +      │  • reset()       │
│             │    Reward + Done      │  • state()       │
└─────────────┘                      └──────┬───────────┘
                                            │
                              ┌─────────────┼─────────────┐
                              ▼             ▼             ▼
                        ┌──────────┐  ┌──────────┐  ┌──────────┐
                        │  Tools   │  │ Customer │  │ Rewards  │
                        │ Registry │  │ Simulator│  │ Engine   │
                        └──────────┘  └──────────┘  └──────────┘
```

---

## 🔄 Observation Space

Each step returns an `Observation` with the following fields:

| Field | Type | Description |
|---|---|---|
| `customer_message` | `str` | What the customer just said |
| `emotion_level` | `float` | 0.0 (calm) → 1.0 (furious) |
| `order_info` | `OrderInfo` | Order ID, product, status, amount, dates |
| `conversation_history` | `list[dict]` | Full chat history (`role`, `content`) |
| `available_tools` | `list[ToolSpec]` | Tools with descriptions and parameter specs |
| `last_action_feedback` | `str \| None` | Feedback from the previous action |
| `instructions` | `str` | Plain-English guidance for the agent |
| `turn_number` / `max_turns` | `int` | Current and maximum step count |

---

## 🎮 Action Space

Agents emit a JSON object in one of two formats:

```json
// FORMAT 1: Respond to customer
{"action_type": "respond", "message": "Your empathetic message here"}

// FORMAT 2: Use a tool
{"action_type": "call_tool", "tool_name": "check_order_status", "tool_parameters": {"order_id": "ORD-123"}}
```

### 🔧 Available Tools

| Tool | Parameters | Description |
|---|---|---|
| `check_order_status` | `order_id` | Look up order details (product, status, delivery) |
| `process_refund` | `order_id`, `reason` | Initiate a refund for the order |
| `escalate_to_manager` | `order_id`, `reason` | Escalate case to a human manager |

---

## 📋 Tasks

### Task 1 — 🟢 Easy: Order Lookup

| Property | Value |
|---|---|
| **Scenario** | Calm customer asks about order status |
| **Initial Emotion** | 0.2 |
| **Required Tools** | `check_order_status` |
| **Expected Steps** | 3 |
| **Order** | Wireless Bluetooth Headphones ($79.99) |

### Task 2 — 🟡 Medium: Angry Customer Refund

| Property | Value |
|---|---|
| **Scenario** | Frustrated customer with delayed order wants a refund |
| **Initial Emotion** | 0.7 |
| **Required Tools** | `check_order_status`, `process_refund` |
| **Expected Steps** | 5 |
| **Order** | Smart Fitness Watch ($149.99) |

### Task 3 — 🔴 Hard: Wrong Item + Escalation

| Property | Value |
|---|---|
| **Scenario** | Furious customer received wrong item, needs refund AND manager escalation |
| **Initial Emotion** | 0.85 |
| **Required Tools** | `check_order_status`, `process_refund`, `escalate_to_manager` |
| **Expected Steps** | 7 |
| **Order** | 15-inch Gaming Laptop ($1,299.99) |

---

## 📊 Grading

End-of-episode score (0.0 → 1.0) computed as a **weighted sum** of four sub-scores:

```
final_score = 0.40 × resolution + 0.25 × tool_usage + 0.20 × emotional_iq + 0.15 × efficiency
```

| Sub-score | Weight | What It Measures |
|---|---|---|
| **Resolution** | 40% | Were all success conditions met? (partial credit supported) |
| **Tool Usage** | 25% | Were required tools called correctly? Penalty for unnecessary calls |
| **Emotional IQ** | 20% | Did emotion decrease? Was language empathetic? No rudeness? |
| **Efficiency** | 15% | Completed within expected step count? |

> 💡 All grading is **deterministic** — zero randomness, pure formula over state variables.

---

## 🏆 Dense Reward Signals

Per-step rewards guide the agent throughout the episode:

### Positive Signals

| Signal | Value | Condition |
|---|---|---|
| ✅ Correct tool | +0.20 | Called a required tool for the first time |
| 📈 Progress | +0.15 | Tool executed successfully |
| 💬 Empathy bonus | +0.10 | Response contains empathetic language |
| 🎯 Resolution | +0.30 | All success conditions met |
| ⚡ Efficiency | +0.10 | Completed ≤ expected steps |

### Negative Signals

| Signal | Value | Condition |
|---|---|---|
| ❌ Wrong tool | −0.15 | Called unnecessary/wrong tool |
| 😤 Rude response | −0.20 | Dismissive or rude language detected |
| 🙈 Ignored customer | −0.15 | Used tool without acknowledging customer |
| 🔁 Redundant step | −0.10 | Exact duplicate of previous action |
| 📉 Frustration spike | −0.05 | Customer emotion increased |

### Failure States (Episode Terminates)

| Condition | Trigger | Penalty |
|---|---|---|
| **Max steps** | ≥ 10 steps | −0.30 |
| **Customer rage** | Emotion ≥ 0.95 | −0.40 |
| **Repeated invalid** | 3× bad actions in a row | −0.30 |

---

## 📈 Baseline Scores

Results from the **mock (optimal scripted) agent**:

| Task | Difficulty | Score | Reward | Steps | Status |
|---|---|---|---|---|---|
| `easy` | 🟢 Easy | ~0.95 | ~0.65 | 3 | ✅ RESOLVED |
| `medium` | 🟡 Medium | ~0.90 | ~0.80 | 5 | ✅ RESOLVED |
| `hard` | 🔴 Hard | ~0.85 | ~0.95 | 7 | ✅ RESOLVED |

> _Scores vary for LLM agents based on model capability._

---

## 🔌 API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Reset environment (`{"task_id": "easy"}`) |
| `POST` | `/step` | Execute action (AgentAction body) |
| `GET` | `/state` | Get internal environment state |
| `GET` | `/tasks` | List available tasks with metadata |
| `GET` | `/health` | Health check |

### Example Requests

#### Reset Environment

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "medium"}'
```

#### Execute an Action

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "respond",
    "message": "I am sorry for the inconvenience. Let me look into your order right away."
  }'
```

#### Call a Tool

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "call_tool",
    "tool_name": "check_order_status",
    "tool_parameters": {"order_id": "ORD-20240310-042"}
  }'
```

#### Check State

```bash
curl http://localhost:8000/state
```

#### List Tasks

```bash
curl http://localhost:8000/tasks
```

---

## 🐳 Docker

Build and run the containerized environment:

```bash
# Build the image
docker build -t agentcare-x .

# Run the container
docker run -p 8000:8000 agentcare-x

# Health check
curl http://localhost:8000/health
```

---

## 🧪 Testing

### Run All Tasks (Mock Mode)

```bash
python inference.py --mock
```

### Run with a Specific LLM

```bash
# Example: Meta Llama 3 via Hugging Face Inference API
set API_BASE_URL=https://api-inference.huggingface.co/v1
set MODEL_NAME=meta-llama/Llama-3-8b-instruct
set HF_TOKEN=hf_your_token_here
python inference.py
```

### Run with OpenAI

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4
set HF_TOKEN=sk-your-openai-key
python inference.py
```

### Run with Ollama (Local)

```bash
set API_BASE_URL=http://localhost:11434/v1
set MODEL_NAME=llama3
set HF_TOKEN=no-key
python inference.py
```

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/new-task`
3. **Add** your task to `tasks/` following the existing pattern
4. **Test** with `python inference.py --mock`
5. **Submit** a pull request

### Adding a New Task

Create a new file in `tasks/` following this template:

```python
TASK: dict = {
    "task_id": "your_task_id",
    "difficulty": "easy|medium|hard",
    "description": "Describe the scenario...",
    "initial_emotion": 0.5,          # 0.0 → 1.0
    "max_steps": 10,
    "required_tools": ["check_order_status"],
    "expected_steps": 4,
    "success_conditions": [
        "tool:check_order_status called",
        "emotion_reduced",
    ],
    "order_data": {
        "order_id": "ORD-YYYYMMDD-XXX",
        "product": "Product Name",
        "status": "shipped|delivered|wrong_item|delayed",
        "amount": 99.99,
        "order_date": "YYYY-MM-DD",
        "estimated_delivery": "YYYY-MM-DD",
    },
}
```

Then register it in `env/environment.py` and `inference.py`.

---

## 📄 License

MIT License — built for the [OpenEnv](https://github.com/openenv) ecosystem.

---

<div align="center">

**Built with ❤️ for the Meta Hackathon**

*AgentCare X — Because AI agents should be smart AND empathetic.*

</div>
]]>
