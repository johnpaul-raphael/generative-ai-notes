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

if not api_key:
    raise ValueError("OPENAI_API_KEY not set in .env")


# ============================================================================
# SECTION 1: Basic Chain (Simple Prompt + Model + Parser)
# ============================================================================
print("=" * 70)
print("SECTION 1: Basic Chain - Loan Summary")
print("=" * 70)

open_ai = ChatOpenAI(api_key=api_key, model=model_name, temperature=0.3)
prompt = ChatPromptTemplate.from_template(
    "You are a credit analyst. Summarise this loan note in ONE line:\n\n{note}"
)
chain = prompt | open_ai | StrOutputParser()

result = chain.invoke(
    {"note": "Credit score 762, FOIR 33%, no defaults, salaried, 8 yrs experience"}
)
print(f"Summary: {result}\n")


# ============================================================================
# SECTION 2: Tools for Agent to Use
# ============================================================================
print("=" * 70)
print("SECTION 2: Define Tools")
print("=" * 70)


@tool
def get_credit_score(pan: str) -> int:
    """Fetch the CIBIL credit score for a customer using their PAN number.
    Use this whenever a credit decision needs a bureau score."""
    fake_bureau = {"ABCDE1234F": 762, "XYZAB9876K": 640}
    score = fake_bureau.get(pan, 700)
    print(f"  [get_credit_score] PAN={pan} -> Score={score}")
    return score


@tool
def calculate_foir(monthly_income: float, existing_emi: float) -> float:
    """Calculate FOIR (Fixed Obligation to Income Ratio) as a percentage.
    Use this to check whether the applicant's existing EMIs are within bank policy."""
    foir = round((existing_emi / monthly_income) * 100, 1)
    print(f"  [calculate_foir] Income={monthly_income}, EMI={existing_emi} -> FOIR={foir}%")
    return foir


print("Tools defined: get_credit_score, calculate_foir\n")


# ============================================================================
# SECTION 3: LangGraph Workflow (StateGraph + Router)
# ============================================================================
print("=" * 70)
print("SECTION 3: LangGraph Loan Eligibility Workflow")
print("=" * 70)


class ParsedApplication(TypedDict):
    """Structured fields extracted from free-text application."""
    pan: str
    monthly_income: float
    existing_emi: float


class LoanState(TypedDict):
    """Shared state throughout the graph execution."""
    prompt: str
    pan: str
    monthly_income: float
    emi_outstanding: float
    credit_score: int
    foir: float
    decision: str
    reason: str


# ---------- NODES ----------
parser_model = ChatOpenAI(api_key=api_key, model=model_name, temperature=0)
structured_parser = parser_model.with_structured_output(ParsedApplication)


def parse_application(state: LoanState) -> dict:
    """LLM node: extract structured fields from free-text prompt."""
    parsed: ParsedApplication = structured_parser.invoke(
        "Extract the PAN number, monthly income, and existing EMI from this loan "
        f"application request:\n\n{state['prompt']}"
    )
    print(f"[parse_application] Extracted: PAN={parsed['pan']}, "
          f"Income={parsed['monthly_income']}, EMI={parsed['existing_emi']}")
    return {
        "pan": parsed["pan"],
        "monthly_income": parsed["monthly_income"],
        "emi_outstanding": parsed["existing_emi"],
    }


def fetch_credit_score(state: LoanState) -> dict:
    """Fetch credit score from simulated bureau."""
    score = get_credit_score.invoke({"pan": state["pan"]})
    return {"credit_score": score}


def calculate_foir_node(state: LoanState) -> dict:
    """Calculate FOIR ratio."""
    foir = calculate_foir.invoke({
        "monthly_income": state["monthly_income"],
        "existing_emi": state["emi_outstanding"]
    })
    return {"foir": foir}


def decide(state: LoanState) -> dict:
    """Bank policy decision: deterministic, not LLM-driven."""
    if state["credit_score"] < 700:
        return {
            "decision": "REJECT",
            "reason": f"Credit score {state['credit_score']} below cut-off 700"
        }
    if state["foir"] > 50:
        return {
            "decision": "REJECT",
            "reason": f"FOIR {state['foir']}% exceeds policy cap 50%"
        }
    return {
        "decision": "APPROVE",
        "reason": f"Score {state['credit_score']}, FOIR {state['foir']}% within policy"
    }


def approve_note(state: LoanState) -> dict:
    """Log approval."""
    print(f"✓ [APPROVED] {state['pan']} — {state['reason']}")
    return {}


def reject_note(state: LoanState) -> dict:
    """Log rejection."""
    print(f"✗ [REJECTED] {state['pan']} — {state['reason']}")
    return {}


# ---------- ROUTER ----------
def route_decision(state: LoanState) -> Literal["approve_note", "reject_note"]:
    """Route to approval or rejection node based on decision."""
    return "approve_note" if state["decision"] == "APPROVE" else "reject_note"


# ---------- BUILD GRAPH ----------
builder = StateGraph(LoanState)

# Add all nodes
builder.add_node("parse_application", parse_application)
builder.add_node("fetch_credit_score", fetch_credit_score)
builder.add_node("calculate_foir", calculate_foir_node)
builder.add_node("decide", decide)
builder.add_node("approve_note", approve_note)
builder.add_node("reject_note", reject_note)

# Linear edges: parse → fetch → calc → decide
builder.add_edge(START, "parse_application")
builder.add_edge("parse_application", "fetch_credit_score")
builder.add_edge("fetch_credit_score", "calculate_foir")
builder.add_edge("calculate_foir", "decide")

# Conditional routing based on decision
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

# Compile with memory
graph = builder.compile(checkpointer=InMemorySaver())
print("Graph compiled successfully.\n")


# ============================================================================
# SECTION 4: Test the Workflow
# ============================================================================
print("=" * 70)
print("SECTION 4: Testing Workflow with Sample Inputs")
print("=" * 70)

test_cases = [
    {
        "name": "Test Case 1: Good applicant (should APPROVE)",
        "prompt": "PAN ABCDE1234F, monthly income 90000, existing EMI 30000. Eligible?"
    },
    {
        "name": "Test Case 2: Low score (should REJECT)",
        "prompt": "PAN XYZAB9876K, monthly income 50000, existing EMI 10000. Eligible?"
    },
    {
        "name": "Test Case 3: High FOIR (should REJECT)",
        "prompt": "PAN ABCDE1234F, monthly income 50000, existing EMI 40000. Eligible?"
    }
]

for test in test_cases:
    print(f"\n{test['name']}")
    print("-" * 70)

    config = {"configurable": {"thread_id": f"test-{test['name'][:5]}"}}

    result = graph.invoke(
        {"prompt": test["prompt"], "pan": "", "monthly_income": 0,
         "emi_outstanding": 0, "credit_score": 0, "foir": 0,
         "decision": "", "reason": ""},
        config
    )

    print(f"Final Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)
