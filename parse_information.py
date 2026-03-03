import re
from pathlib import Path
import pandas as pd

STEP_FINAL_RE = re.compile(r"=+\s*STEP\s+(\d+)\s+—\s+FINAL DECISION\s*=+", re.UNICODE)

# Grab the block from "This step's negotiation transcript:" until the blank line before "Previous outcome"
NEGOTIATION_RE = re.compile(
    r"This step's negotiation transcript:\s*\n(.*?)\n\nPrevious outcome",
    re.DOTALL | re.UNICODE
)

# Current positions inside each agent's LLM INPUT
# We’ll parse them once per step (from A1 input if present, else fallback to anywhere in step)
POSITIONS_RE = re.compile(
    r"Current Positions:\s*\n-+\s*A1\s+is\s+at\s+\((\d+),\s*(\d+)\)\s*\n-+\s*A2\s+is\s+at\s+\((\d+),\s*(\d+)\)",
    re.UNICODE
)

# Reasoning string inside RAW OUTPUT OBJECT:
# reasoning="...." (handle escaped quotes \" and newlines)
def extract_reasoning(step_text: str, agent: str) -> str:
    # Find the agent section first to avoid accidentally grabbing the other agent's reasoning
    # We anchor from "STEP X — A1 DECISION — RAW OUTPUT OBJECT" to next "STEP X — A1 DECISION — RAW TEXT OUTPUT"
    sect_re = re.compile(
        rf"STEP\s+\d+\s+—\s+{agent}\s+DECISION\s+—\s+RAW OUTPUT OBJECT\s*\n(.*?)\n\nSTEP\s+\d+\s+—\s+{agent}\s+DECISION\s+—\s+RAW TEXT OUTPUT",
        re.DOTALL | re.UNICODE
    )
    m = sect_re.search(step_text)
    if not m:
        return ""

    sect = m.group(1)

    # reasoning="..."; allow escaped chars
    # This is conservative: it finds reasoning=" then captures until the next unescaped "
    rm = re.search(r'reasoning="((?:\\.|[^"\\])*)"', sect, re.DOTALL)
    if not rm:
        return ""

    raw = rm.group(1)
    # unescape common sequences
    raw = raw.replace(r"\n", "\n").replace(r"\\", "\\").replace(r"\"", '"')
    return raw.strip()

EXECUTION_RE = re.compile(
    r"=+\s*STEP\s+(\d+)\s+—\s+ENVIRONMENT\s*=+.*?\n---\s*EXECUTION RESULT\s*---\s*\n(.*?)(?=\n\n=+|\Z)",
    re.DOTALL | re.UNICODE
)

def split_steps(text: str):
    """Return list of (step_number, step_final_block_text)."""
    starts = [(m.start(), int(m.group(1))) for m in STEP_FINAL_RE.finditer(text)]
    blocks = []
    for i, (pos, step) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        blocks.append((step, text[pos:end]))
    return blocks

def parse_log(path: str) -> pd.DataFrame:
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # Build execution lookup by step
    exec_map = {}
    for m in EXECUTION_RE.finditer(text):
        step = int(m.group(1))
        exec_map[step] = m.group(2).strip()

    rows = []
    for step, step_block in split_steps(text):
        # positions
        pm = POSITIONS_RE.search(step_block)
        if pm:
            a1 = f"({pm.group(1)},{pm.group(2)})"
            a2 = f"({pm.group(3)},{pm.group(4)})"
            pos_cell = f"Step {step} | A1={a1} | A2={a2}"
        else:
            pos_cell = f"Step {step} | A1=? | A2=?"

        # negotiation block
        nm = NEGOTIATION_RE.search(step_block)
        negotiation = nm.group(1).strip() if nm else ""

        # execution outcome for the same step
        execution = exec_map.get(step, "")

        # reasoning
        a1_reason = extract_reasoning(step_block, "A1")
        a2_reason = extract_reasoning(step_block, "A2")
        reasoning_cell = ""
        if a1_reason or a2_reason:
            reasoning_cell = f"A1 reasoning:\n{a1_reason}\n\nA2 reasoning:\n{a2_reason}".strip()

        rows.append({
            "Step + current positions": pos_cell,
            "Negotiation (6 messages)": negotiation,
            "Execution outcome": execution,
            "Reasoning (A1 + A2)": reasoning_cell,
        })

    df = pd.DataFrame(rows)

    # save outputs
    out_csv = Path(path).with_suffix(".parsed.csv")
    out_xlsx = Path(path).with_suffix(".parsed.xlsx")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    df.to_excel(out_xlsx, index=False)

    return df

if __name__ == "__main__":
    # change this to your file
    infile = "simulation_log_1.txt"
    df = parse_log(infile)
    print(df.head(10).to_string(index=False))
    print("\nSaved:", Path(infile).with_suffix(".parsed.csv"), "and", Path(infile).with_suffix(".parsed.xlsx"))