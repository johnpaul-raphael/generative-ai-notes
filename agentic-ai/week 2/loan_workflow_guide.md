# LangGraph Loan Eligibility Workflow - Complete Guide

## Overview
This guide walks through a complete agentic loan eligibility workflow using LangGraph, LangChain, and OpenAI's LLM.

---

## Issues Found & Fixed in Your Notebook

### 1. **Typo in Node Name** (Cell `1f5d2a5f`)
```python
# ❌ WRONG
builder.add_node("pase_application", parse_application)

# ✅ CORRECT
builder.add_node("parse_application", parse_application)
```

### 2. **Incomplete Conditional Edges**
```python
# ❌ WRONG - missing "reject" path
builder.add_conditional_edges("decide", route,
                            {"approve": "approve_note"})

# ✅ CORRECT - both paths defined
builder.add_conditional_edges(
    "decide",
    route_decision,
    {
        "approve_note": "approve_note",
        "reject_note": "reject_note"
    }
)
```

### 3. **Router Returns Wrong Values**
```python
# ❌ WRONG - returns "approve" but edge expects "approve_note"
def route(state: LoanState) -> Literal["approve", "reject"]:
    return "approve" if state["decision"] == "APPROVE" else "reject"

# ✅ CORRECT - returns exact node names
def route_decision(state: LoanState) -> Literal["approve_note", "reject_note"]:
    return "approve_note" if state["decision"] == "APPROVE" else "reject_note"
```

### 4. **Missing Terminal Edges**
```python
# ❌ WRONG - no END edges, graph hangs
builder.add_conditional_edges("decide", route, {"approve": "approve_note"})

# ✅ CORRECT - edges lead to END
builder.add_edge("approve_note", END)
builder.add_edge("reject_note", END)
```

### 5. **Missing Graph Compilation**
```python
# ❌ INCOMPLETE - never compiled
builder = StateGraph(LoanState)
# ... add nodes ...

# ✅ CORRECT
graph = builder.compile(checkpointer=InMemorySaver())
```

---

## Complete Corrected Workflow

### Cell 1: Imports & Setup
```python
import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPEN_AI_MODEL", "gpt-4o-mini")
```

### Cell 2: Define State & Nodes
```python
class LoanState(TypedDict):
    prompt: str
    pan: str
    monthly_income: float
    emi_outstanding: float
    credit_score: int
    foir: float
    decision: str
    reason: str


class ParsedApplication(TypedDict):
    pan: str
    monthly_income: float
    existing_emi: float


parser_model = ChatOpenAI(api_key=api_key, model=model_name, temperature=0)
structured_parser = parser_model.with_structured_output(ParsedApplication)


def parse_application(state: LoanState) -> dict:
    parsed: ParsedApplication = structured_parser.invoke(
        "Extract PAN, monthly income, existing EMI:\n\n" + state['prompt']
    )
    return {
        "pan": parsed["pan"],
        "monthly_income": parsed["monthly_income"],
        "emi_outstanding": parsed["existing_emi"],
    }


def fetch_credit_score(state: LoanState) -> dict:
    fake_bureau = {"ABCDE1234F": 762, "XYZAB9876K": 640}
    score = fake_bureau.get(state["pan"], 700)
    return {"credit_score": score}


def calculate_foir(state: LoanState) -> dict:
    foir = round((state["emi_outstanding"] / state["monthly_income"]) * 100, 1)
    return {"foir": foir}


def decide(state: LoanState) -> dict:
    if state["credit_score"] < 700:
        return {
            "decision": "REJECT",
            "reason": f"Credit score {state['credit_score']} < 700"
        }
    if state["foir"] > 50:
        return {
            "decision": "REJECT",
            "reason": f"FOIR {state['foir']}% > 50%"
        }
    return {
        "decision": "APPROVE",
        "reason": f"Score {state['credit_score']}, FOIR {state['foir']}% OK"
    }


def approve_note(state: LoanState) -> dict:
    print(f"✓ APPROVED: {state['pan']} — {state['reason']}")
    return {}


def reject_note(state: LoanState) -> dict:
    print(f"✗ REJECTED: {state['pan']} — {state['reason']}")
    return {}
```

### Cell 3: Router Function
```python
def route_decision(state: LoanState) -> Literal["approve_note", "reject_note"]:
    return "approve_note" if state["decision"] == "APPROVE" else "reject_note"
```

### Cell 4: Build & Compile Graph
```python
builder = StateGraph(LoanState)

# Add nodes
builder.add_node("parse_application", parse_application)
builder.add_node("fetch_credit_score", fetch_credit_score)
builder.add_node("calculate_foir", calculate_foir)
builder.add_node("decide", decide)
builder.add_node("approve_note", approve_note)
builder.add_node("reject_note", reject_note)

# Linear pipeline
builder.add_edge(START, "parse_application")
builder.add_edge("parse_application", "fetch_credit_score")
builder.add_edge("fetch_credit_score", "calculate_foir")
builder.add_edge("calculate_foir", "decide")

# Conditional routing
builder.add_conditional_edges(
    "decide",
    route_decision,
    {
        "approve_note": "approve_note",
        "reject_note": "reject_note"
    }
)

# Terminal edges
builder.add_edge("approve_note", END)
builder.add_edge("reject_note", END)

# Compile
graph = builder.compile(checkpointer=InMemorySaver())
```

### Cell 5: Test Workflow
```python
config = {"configurable": {"thread_id": "test-001"}}

result = graph.invoke(
    {
        "prompt": "PAN ABCDE1234F, monthly income 90000, existing EMI 30000. Eligible?",
        "pan": "", "monthly_income": 0, "emi_outstanding": 0,
        "credit_score": 0, "foir": 0, "decision": "", "reason": ""
    },
    config
)

print(f"Decision: {result['decision']}")
print(f"Reason: {result['reason']}")
```

---

## Graph Flow Diagram

```
START
  ↓
[parse_application] → Extract PAN, Income, EMI
  ↓
[fetch_credit_score] → Look up CIBIL score
  ↓
[calculate_foir] → Calculate EMI/Income ratio
  ↓
[decide] → Apply bank policy rules
  ↓
  ├─→ "APPROVE" ─→ [approve_note] → END
  │
  └─→ "REJECT"  ─→ [reject_note]  → END
```

---

## Key Concepts

### StateGraph
- Defines the state shape (TypedDict) that flows through the graph
- Each node reads from state and returns a partial dict to merge

### Nodes
- Pure functions that take state in, return dict to merge
- Can be LLM-based (use chains) or deterministic (business logic)

### Edges
- **Linear edges**: `add_edge("A", "B")` — always go from A to B
- **Conditional edges**: `add_conditional_edges("A", router, {"path1": "B", "path2": "C"})`
  - Router function inspects state and decides which edge to take

### Router
- Must return a string matching one of the edge keys
- This example: returns `"approve_note"` or `"reject_note"` based on decision

### Checkpointer
- `InMemorySaver()` for development (loses history on restart)
- For production, use `PostgresSaver` or similar for persistence

---

## Next Steps

1. **Test with different applicants** in cell 5
2. **Add more business rules** to the `decide()` node (e.g., age, employment type)
3. **Replace fake_bureau** with a real API call
4. **Add logging/observability** to track decisions over time
5. **Deploy** using LangServe or FastAPI wrapper
