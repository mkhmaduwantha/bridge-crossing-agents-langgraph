from typing import TypedDict, List, Dict, Tuple, Optional
from langgraph.graph import StateGraph, END
from openai import OpenAI
import re
import traceback

# ==========================================
# LOGGING SETUP
# ==========================================

LOG_FILE = "simulation_log.txt"

def log_write(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

open(LOG_FILE, "w").close()

# ==========================================
# LLM SETUP
# ==========================================

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="sk-lm-CnAy7yZu:OFT986jcpFQgYmA5hOxg"
)

def llm_call(prompt: str, tag: str):
    log_write(f"\n==============================")
    log_write(f"{tag} — LLM INPUT")
    log_write(f"==============================")
    log_write(prompt)

    try:
        response = client.chat.completions.create(
            model="google/gemma-3-4b",
            messages=[
                {"role": "system", "content": "You are a rational strategic agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        raw_output = response.choices[0].message.content

        log_write(f"\n{tag} — RAW OUTPUT OBJECT")
        log_write(str(response))

        log_write(f"\n{tag} — RAW TEXT OUTPUT")
        log_write(f"[Length: {len(raw_output) if raw_output else 0}]")
        log_write(raw_output if raw_output else "[EMPTY STRING RETURNED]")

        return raw_output.strip() if raw_output else ""

    except Exception as e:
        log_write(f"\n{tag} — EXCEPTION")
        log_write(str(e))
        log_write(traceback.format_exc())
        return ""

# ==========================================
# GRID CONFIG
# ==========================================

GRID_W, GRID_H = 5, 5
WALLS = {(2, 4), (2, 2), (2, 0)}
BRIDGES = {"upper bridge": (2, 3), "lower bridge": (2, 1)}

A1_START = (0, 2)
A2_START = (4, 2)

A1_GOAL_X = 4
A2_GOAL_X = 0

MOVE_DELTAS = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "STAY": (0, 0),
}

# ==========================================
# STATE TYPE
# ==========================================

class SimulationState(TypedDict):
    grid_state: Dict[str, Tuple[int, int]]
    transcript: str
    shared_history: List[str]
    outcome_summary: str
    step: int
    move1: Optional[str]
    move2: Optional[str]
    recent_rounds: List[Dict[str, str]]   # NEW: keep last N rounds

# ==========================================
# DESCRIPTIVE GRID EXPLANATION
# ==========================================

def describe_grid(state):
    return f"""
The world is a square grid that is 5 columns wide and 5 rows tall.

Coordinates:
- x increases as you move RIGHT.
- y increases as you move UP.

The middle column (x = 2) represents a river that runs vertically
from the bottom of the grid to the top.

Most cells in this river are water and cannot be stepped on.
The water cells are located at: {sorted(list(WALLS))}

There are exactly two bridges that allow crossing the river:

1. The upper bridge located at {BRIDGES["upper bridge"]}
2. The lower bridge located at {BRIDGES["lower bridge"]}

Only ONE agent may stand on a bridge at any time.

Movement Rules:
- Each turn both agents move simultaneously.
- You may move one step: UP, DOWN, LEFT, RIGHT, or STAY.
- You cannot move outside the grid boundaries.
- You cannot enter water cells.
- You cannot occupy the same cell as the other agent.
- You cannot swap positions in one move.
- If a move is invalid, you remain in place.

Current Positions:
- A1 is at {state['grid_state']['A1']}
- A2 is at {state['grid_state']['A2']}

Goals:
- A1 must reach any cell where x = {A1_GOAL_X}
- A2 must reach any cell where x = {A2_GOAL_X}
"""

def describe_grid_for_agent(state, agent_name):

    pos = state["grid_state"][agent_name]

    if agent_name == "A1":
        goal_x = A1_GOAL_X
        direction_hint = "RIGHT"
        wrong_hint = "LEFT"
    else:
        goal_x = A2_GOAL_X
        direction_hint = "LEFT"
        wrong_hint = "RIGHT"

    distance = abs(goal_x - pos[0])

    return f"""
The world is a square grid that is 5 columns wide and 5 rows tall.

Coordinates:
- x increases when moving RIGHT
- x decreases when moving LEFT
- y increases when moving UP

The middle column (x = 2) represents a river that runs vertically
from the bottom of the grid to the top.

Most cells in this river are water and cannot be stepped on.
The water cells are located at: {sorted(list(WALLS))}

There are exactly two bridges that allow crossing the river:

1. The upper bridge located at {BRIDGES["upper bridge"]}
2. The lower bridge located at {BRIDGES["lower bridge"]}

Only ONE agent may stand on a bridge at any time.

Except for (1) water cells and (2) the other agent’s current cell, every other coordinate inside the 5×5 grid is walkable.

YOUR IDENTITY:
You are {agent_name}.
Your current position is {pos}.

YOUR GOAL:
Your goal is to reach ANY cell where x = {goal_x}.

IMPORTANT DIRECTIONAL FACT:
- Moving {direction_hint} reduces your distance to the goal.
- Moving {wrong_hint} increases your distance from the goal.

Your current horizontal distance to goal is: {distance}

Movement Rules:
- Each turn both agents move simultaneously.
- You may move one step: UP, DOWN, LEFT, RIGHT, or STAY.
- You cannot move outside the grid boundaries.
- You cannot enter water cells.
- You cannot occupy the same cell as the other agent.
- You cannot swap positions in one move.
- If a move is invalid, you remain in place.

Current Positions:
- A1 is at {state['grid_state']['A1']}
- A2 is at {state['grid_state']['A2']}

GOALS (NOT SYMMETRIC):
- A1 goal: reach x=4 (right edge)
- A2 goal: reach x=0 (left edge)
- Your goal is the horizontal opposite of the other agent’s goal.
"""

# ==========================================
# HELPERS
# ==========================================

def apply_move(pos, move):
    dx, dy = MOVE_DELTAS.get(move, (0, 0))
    return (pos[0] + dx, pos[1] + dy)

def valid_env(pos):
    return 0 <= pos[0] < GRID_W and 0 <= pos[1] < GRID_H and pos not in WALLS

def detect_collision(a1_new, a2_new, a1_old, a2_old):
    if a1_new == a2_new:
        return "same_cell"
    if a1_new == a2_old and a2_new == a1_old:
        return "swap"
    for bridge in BRIDGES.values():
        if a1_new == bridge and a2_new == bridge:
            return "same_bridge"
    return None

def parse_move(text):
    match = re.search(r"(UP|DOWN|LEFT|RIGHT|STAY)", text.upper())
    return match.group(1) if match else "STAY"

COLLISION_DESCRIPTIONS = {
    "same_cell": "Invalid: both agents tried to end on the same cell.",
    "swap": "Invalid: agents tried to swap positions in the same turn (A1->A2_old and A2->A1_old).",
    "same_bridge": "Invalid: both agents tried to use the same bridge in the same turn (only one agent allowed on a bridge).",
}

ENV_INVALID_DESCRIPTIONS = {
    "out_of_bounds": "Invalid: you cannot move outside the grid boundaries.",
    "water": "Invalid: you cannot step onto water (river) cells; only bridge cells allow crossing.",
}

def env_invalid_reason(pos):
    """Return (reason_code or None)."""
    x, y = pos
    if not (0 <= x < GRID_W and 0 <= y < GRID_H):
        return "out_of_bounds"
    if pos in WALLS:
        return "water"
    return None

def describe_env_effect(agent_name, intended_move, old_pos, attempted_pos, final_pos, reason_code):
    if reason_code is None:
        return f"{agent_name} moved {intended_move} from {old_pos} to {final_pos}."
    # It attempted something invalid and therefore stayed
    return (
        f"{agent_name} tried {intended_move} from {old_pos} to {attempted_pos}, "
        f"but stayed at {final_pos}. Reason: {ENV_INVALID_DESCRIPTIONS.get(reason_code, reason_code)}"
    )

def format_recent_rounds(state, n=3):
    rounds = state.get("recent_rounds", [])
    if not rounds:
        return "(none yet)"
    last = rounds[-n:]
    out = []
    for r in last:
        out.append(
            f"- Step {r['step']}:\n"
            f"  Negotiation:\n{r['transcript'].rstrip()}\n"
            f"  Final decisions: A1={r['move1']} | A2={r['move2']}\n"
            f"  Outcome: {r['outcome'].replace(chr(10),' ')}"
        )
    return "\n".join(out)

# ==========================================
# NODE 1 — NEGOTIATION
# ==========================================

def negotiation_node(state: SimulationState):

    log_write(f"\n\n========== STEP {state['step']} — NEGOTIATION ==========")

    transcript = ""
    # grid_desc = describe_grid(state)
    
    recent = format_recent_rounds(state, n=3)

    for i in range(6):
        speaker = "A1" if i % 2 == 0 else "A2"
        grid_desc = describe_grid_for_agent(state, speaker)
        prompt = f"""
You are {speaker}.

Your goal is to cross the river and reach your target side.

{grid_desc}

Last 3 rounds (negotiations + final decisions + outcomes):
{recent}

Previous outcome:
{state['outcome_summary'] if state['outcome_summary'] else "(none yet)"}

Negotiation so far this step:
{transcript if transcript else "(none yet)"}

You must propose your intended move AND send a short message.

Respond EXACTLY in this format:

PROPOSED_MOVE: <UP|DOWN|LEFT|RIGHT|STAY>
MESSAGE: <optional arbitraty message to other agent>
"""

        response = llm_call(prompt, f"STEP {state['step']} — {speaker} NEGOTIATION")

        proposed_move = parse_move(response)
        message = response.split("MESSAGE:")[-1].strip() if "MESSAGE:" in response else ""

        transcript += f"{speaker}: PROPOSED_MOVE={proposed_move} | MESSAGE={message}\n"

    log_write("\n--- NEGOTIATION TRANSCRIPT ---")
    log_write(transcript)

    state["transcript"] = transcript
    return state

# ==========================================
# NODE 2 — FINAL DECISION
# ==========================================

def decision_node(state: SimulationState):

    log_write(f"\n========== STEP {state['step']} — FINAL DECISION ==========")

    # grid_desc = describe_grid(state)
    recent = format_recent_rounds(state, n=3)
    def decide(agent_name):
        grid_desc = describe_grid_for_agent(state, agent_name)
        prompt = f"""
You are {agent_name}.

After reviewing the negotiation transcript below,
you must now commit to your FINAL MOVE.

{grid_desc}

Last 3 rounds (negotiations + final decisions + outcomes):
{recent}

This step's negotiation transcript:
{state['transcript']}

Previous outcome (last step):
{state['outcome_summary'] if state['outcome_summary'] else "(none yet)"}

Commit to your FINAL MOVE.

Respond ONLY:
MOVE: <UP|DOWN|LEFT|RIGHT|STAY>
"""
        response = llm_call(prompt, f"STEP {state['step']} — {agent_name} DECISION")
        return parse_move(response)

    state["move1"] = decide("A1")
    state["move2"] = decide("A2")

    return state

# ==========================================
# NODE 3 — ENVIRONMENT
# ==========================================

def environment_node(state: SimulationState):

    log_write(f"\n========== STEP {state['step']} — ENVIRONMENT ==========")

    a1_old = state["grid_state"]["A1"]
    a2_old = state["grid_state"]["A2"]

    move1 = state["move1"] or "STAY"
    move2 = state["move2"] or "STAY"

    # Attempted positions (what they tried)
    a1_attempt = apply_move(a1_old, move1)
    a2_attempt = apply_move(a2_old, move2)

    # Environment validity checks (water/outside)
    a1_reason = env_invalid_reason(a1_attempt)
    a2_reason = env_invalid_reason(a2_attempt)

    # Proposed positions after env constraints (invalid => stay)
    a1_proposed = a1_old if a1_reason else a1_attempt
    a2_proposed = a2_old if a2_reason else a2_attempt

    # Agent-agent collision checks on env-valid proposals
    collision = detect_collision(a1_proposed, a2_proposed, a1_old, a2_old)

    # If agent collision, both stay (and we describe it)
    if collision:
        a1_final, a2_final = a1_old, a2_old
        collision_desc = COLLISION_DESCRIPTIONS.get(collision, collision)
    else:
        a1_final, a2_final = a1_proposed, a2_proposed
        collision_desc = "none"

    state["grid_state"]["A1"] = a1_final
    state["grid_state"]["A2"] = a2_final

    # Descriptive per-agent outcome lines (include env reasons if any)
    a1_line = describe_env_effect("A1", move1, a1_old, a1_attempt, a1_final, a1_reason if not collision else None)
    a2_line = describe_env_effect("A2", move2, a2_old, a2_attempt, a2_final, a2_reason if not collision else None)

    # If agent collision happened, override with collision explanation (they stayed due to collision)
    if collision:
        a1_line = f"A1 intended {move1} from {a1_old} toward {a1_proposed}, but both stayed. Reason: {collision_desc}"
        a2_line = f"A2 intended {move2} from {a2_old} toward {a2_proposed}, but both stayed. Reason: {collision_desc}"

    outcome = f"""
Step {state['step']} RESULT:
{a1_line}
{a2_line}
Agent-agent collision: {collision_desc}
""".strip()

    log_write("\n--- EXECUTION RESULT ---")
    log_write(outcome)

    # Save into shared history
    state["outcome_summary"] = outcome
    state["shared_history"].append(outcome)

    # Save this round into recent_rounds (for next prompts)
    round_record = {
        "step": str(state["step"]),
        "transcript": state.get("transcript", ""),
        "move1": move1,
        "move2": move2,
        "outcome": outcome,
    }
    state["recent_rounds"].append(round_record)
    # Keep it bounded (optional)
    if len(state["recent_rounds"]) > 20:
        state["recent_rounds"] = state["recent_rounds"][-20:]

    state["step"] += 1
    return state

# ==========================================
# LOOP CONDITION
# ==========================================

def should_continue(state: SimulationState):
    if (
        state["grid_state"]["A1"][0] == A1_GOAL_X and
        state["grid_state"]["A2"][0] == A2_GOAL_X
    ):
        log_write("\nSUCCESS: Both agents reached goals.")
        return "end"
    if state["step"] >= 50:
        log_write("\nSTOP: Max steps reached.")
        return "end"
    return "negotiation"

# ==========================================
# BUILD GRAPH
# ==========================================

workflow = StateGraph(SimulationState)

workflow.add_node("negotiation", negotiation_node)
workflow.add_node("decision", decision_node)
workflow.add_node("environment", environment_node)

workflow.set_entry_point("negotiation")
workflow.add_edge("negotiation", "decision")
workflow.add_edge("decision", "environment")

workflow.add_conditional_edges(
    "environment",
    should_continue,
    {"negotiation": "negotiation", "end": END},
)

app = workflow.compile()

# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    initial_state = SimulationState(
    grid_state={"A1": A1_START, "A2": A2_START},
    transcript="",
    shared_history=[],
    outcome_summary="",
    step=0,
    move1=None,
    move2=None,
    recent_rounds=[],  # NEW
    )

    final_state = app.invoke(initial_state)

    print("\nSimulation finished.")
    print("Check simulation_log.txt for full trace.")