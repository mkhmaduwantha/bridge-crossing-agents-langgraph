import json
import re
import pandas as pd
from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from openai import OpenAI

# ============================================================
# LM STUDIO CONFIG
# ============================================================

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="sk-lm-CnAy7yZu:OFT986jcpFQgYmA5hOxg"
)

def llm_call(prompt):
    response = client.chat.completions.create(
        model="google/gemma-3-4b",
        messages=[
            {"role": "system", "content": "You are a rational planning agent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = "agent_trace_log.txt"
open(LOG_FILE, "w").close()  # reset each run

def log_trace(step, agent, prompt, raw):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"STEP: {step} | AGENT: {agent}\n")
        f.write("-"*80 + "\nPROMPT:\n")
        f.write(prompt + "\n")
        f.write("-"*80 + "\nRAW OUTPUT:\n")
        f.write(raw + "\n")


# ============================================================
# GRID CONSTANTS
# ============================================================

GRID_SIZE = 5

WALLS = {(2,0), (2,2), (2,4)}
BRIDGES = {(2,1), (2,3)}

START_POS = {
    "A1": (0,2),
    "A2": (4,2)
}

GOALS = {
    "A1": 4,
    "A2": 0
}


# ============================================================
# STATE
# ============================================================

class GridState(TypedDict, total=False):
    step: int
    agent_positions: Dict[str, tuple]
    history: List[Dict]
    done: bool

    conversation_round: int
    messages: List[Dict]
    final_decisions: Dict[str, str]


# ============================================================
# MOVE LOGIC
# ============================================================

def compute_move(position, move):
    x, y = position
    if move == "UP":
        return (x, y-1)
    if move == "DOWN":
        return (x, y+1)
    if move == "LEFT":
        return (x-1, y)
    if move == "RIGHT":
        return (x+1, y)
    return position

def valid_position(pos):
    x, y = pos
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        return False
    if pos in WALLS:
        return False
    return True


# ============================================================
# NEGOTIATION NODE
# ============================================================

def negotiation_node(agent_name):

    def node(state: GridState):

        position = state["agent_positions"][agent_name]
        other = "A1" if agent_name == "A2" else "A2"

        prompt = f"""
You are {agent_name}.

Grid 5x5. Top-left = (0,0).
Walls: {WALLS}
Bridges: {BRIDGES}
Only one agent may occupy a bridge at a time.

Your position: {position}
Other position: {state['agent_positions'][other]}

Conversation so far:
{state.get("messages", [])}

Discuss and coordinate a plan.
Do NOT execute yet.

Return ONLY raw JSON:
{{
 "message": "...",
 "proposed_move": "UP/DOWN/LEFT/RIGHT/WAIT"
}}
"""

        raw = llm_call(prompt)
        log_trace(state["step"], agent_name, prompt, raw)

        clean = re.sub(r"```json|```", "", raw).strip()

        try:
            parsed = json.loads(clean)
        except:
            parsed = {"message": "Invalid", "proposed_move": "WAIT"}

        state.setdefault("messages", []).append({
            "agent": agent_name,
            "content": parsed["message"],
            "proposal": parsed["proposed_move"]
        })

        state["conversation_round"] += 1

        return state

    return node


# ============================================================
# EXECUTION NODE
# ============================================================

def execution_node(state: GridState):

    proposals = {}

    for msg in reversed(state["messages"]):
        if msg["agent"] not in proposals:
            proposals[msg["agent"]] = msg["proposal"]
        if len(proposals) == 2:
            break

    state["final_decisions"] = proposals
    return state


# ============================================================
# ENVIRONMENT NODE
# ============================================================

def environment_node(state: GridState):

    decisions = state["final_decisions"]

    new_positions = {}
    collisions = {}

    for agent, move in decisions.items():
        old = state["agent_positions"][agent]
        new = compute_move(old, move)

        collision = None

        if not valid_position(new):
            new = old
            collision = "wall_or_boundary"

        new_positions[agent] = new
        collisions[agent] = collision

    # Same cell collision
    if len(set(new_positions.values())) < 2:
        for agent in new_positions:
            new_positions[agent] = state["agent_positions"][agent]
            collisions[agent] = "agent_collision"

    # Bridge conflict
    bridge_occupants = [pos for pos in new_positions.values() if pos in BRIDGES]
    if len(bridge_occupants) > 1:
        for agent in new_positions:
            if new_positions[agent] in BRIDGES:
                new_positions[agent] = state["agent_positions"][agent]
                collisions[agent] = "bridge_conflict"

    for agent in ["A1","A2"]:
        state["history"].append({
            "step": state["step"],
            "agent": agent,
            "move": decisions.get(agent,"WAIT"),
            "collision": collisions[agent]
        })

    state["agent_positions"] = new_positions
    state["step"] += 1

    # Reset negotiation
    state["conversation_round"] = 0
    state["messages"] = []
    state["final_decisions"] = {}

    # Check goal
    if new_positions["A1"][0] == GOALS["A1"] or \
       new_positions["A2"][0] == GOALS["A2"]:
        state["done"] = True

    return state


# ============================================================
# ROUTERS
# ============================================================

def conversation_router(state: GridState):
    if state["conversation_round"] >= 6:
        return "EXECUTE"
    return "A2_msg" if state["conversation_round"] % 2 == 1 else "A1_msg"

def continue_simulation(state: GridState):
    if state["done"] or state["step"] > 20:
        return END
    return "A1_msg"


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(GridState)

builder.add_node("A1_msg", negotiation_node("A1"))
builder.add_node("A2_msg", negotiation_node("A2"))
builder.add_node("EXECUTE", execution_node)
builder.add_node("ENV", environment_node)

builder.set_entry_point("A1_msg")

builder.add_conditional_edges("A1_msg", conversation_router)
builder.add_conditional_edges("A2_msg", conversation_router)

builder.add_edge("EXECUTE", "ENV")

builder.add_conditional_edges("ENV", continue_simulation)

graph = builder.compile()


# ============================================================
# RUN
# ============================================================

initial_state: GridState = {
    "step": 0,
    "agent_positions": START_POS.copy(),
    "history": [],
    "done": False,
    "conversation_round": 0,
    "messages": [],
    "final_decisions": {}
}

final_state = graph.invoke(initial_state)

df = pd.DataFrame(final_state["history"])
print(df)
print("\nFinal Positions:", final_state["agent_positions"])