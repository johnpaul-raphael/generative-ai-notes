# LangGraph for Intermediate: LLM Integration & Real Agents

## Table of Contents
1. [LLM Integration](#llm-integration)
2. [Tools and Tool Calling](#tools-and-tool-calling)
3. [The Agent Loop Pattern](#the-agent-loop-pattern)
4. [Multi-Step Decision Trees](#multi-step-decision-trees)
5. [State Persistence & Memory](#state-persistence--memory)
6. [Real Example: Loan Eligibility Agent](#real-example-loan-eligibility-agent)

---

## LLM Integration

### Concept: Using LLMs as Nodes
Instead of hardcoded logic, let LLMs (Large Language Models) make decisions and generate content.

### Basic: LLM Node
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-key-here"

class LLMState(TypedDict):
    user_input: str
    response: str

def llm_node(state: LLMState) -> dict:
    """Call OpenAI LLM."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # model.invoke() expects a list of messages or a string
    result = model.invoke(f"Answer this: {state['user_input']}")
    
    # result is a message object, extract text
    return {"response": result.content}

builder = StateGraph(LLMState)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

graph = builder.compile()
result = graph.invoke({"user_input": "What is 2+2?", "response": ""})
print(result["response"])  # "2+2 equals 4."
```

### LLM with Chains
For complex prompts, use LangChain's `ChatPromptTemplate`:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ChainState(TypedDict):
    topic: str
    summary: str

def chain_node(state: ChainState) -> dict:
    """Use a chain for structured processing."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(
        "Summarize this topic in one sentence:\n\n{topic}"
    )
    
    # Create a chain: prompt → model → parser
    chain = prompt | model | StrOutputParser()
    
    # Invoke the chain
    result = chain.invoke({"topic": state["topic"]})
    
    return {"summary": result}

builder = StateGraph(ChainState)
builder.add_node("chain", chain_node)
builder.add_edge(START, "chain")
builder.add_edge("chain", END)

graph = builder.compile()
result = graph.invoke({"topic": "Photosynthesis", "summary": ""})
print(result["summary"])
```

---

## Tools and Tool Calling

### What are Tools?
**Tools** are functions that an LLM can call to get information or perform actions. The LLM decides when to use which tool.

### Define a Tool
```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    # Simulated weather data
    weather_data = {"New York": "Sunny", "London": "Rainy", "Paris": "Cloudy"}
    return weather_data.get(location, "Unknown location")

@tool
def calculate_mortgage(principal: float, rate: float, years: int) -> float:
    """Calculate monthly mortgage payment."""
    monthly_rate = rate / 100 / 12
    num_payments = years * 12
    if monthly_rate == 0:
        return principal / num_payments
    payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / \
              ((1 + monthly_rate) ** num_payments - 1)
    return round(payment, 2)

# Tools have automatic docstring extraction for LLM understanding
print(get_weather.description)
# Output: "Get the weather for a location."

print(calculate_mortgage.args)
# Output: {'principal': {...}, 'rate': {...}, 'years': {...}}
```

### Tool Use in Agents

LangGraph provides `create_agent` for tool-using workflows:

```python
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

@tool
def get_credit_score(pan: str) -> int:
    """Get CIBIL credit score using PAN."""
    scores = {"ABCDE1234F": 762, "XYZAB9876K": 640}
    return scores.get(pan, 700)

@tool
def check_income(pan: str) -> float:
    """Get applicant's monthly income."""
    incomes = {"ABCDE1234F": 90000, "XYZAB9876K": 50000}
    return incomes.get(pan, 0)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_credit_score, check_income],
    system_prompt=(
        "You are a loan advisor. Use tools to check credit score and income. "
        "Recommend approval if score > 700 and income > 40000."
    ),
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "user-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Is PAN ABCDE1234F eligible for a loan?"}]},
    config
)

print(result["messages"][-1].content)
# LLM will call both tools, then summarize findings
```

---

## The Agent Loop Pattern

An **agent loop** is: Think → Act → Observe → Repeat until done.

```
START
  ↓
[LLM thinks] → "I need to use tool X"
  ↓
[Call tool X] → Get result
  ↓
[LLM observes] → Decide if more tools needed or done
  ↓
If done: END
If not: loop back to LLM
```

### Manual Agent Loop with LangGraph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia (simulated)."""
    return f"Found info about {query}: ..."

@tool
def fetch_url(url: str) -> str:
    """Fetch web page content."""
    return f"Content of {url}: ..."

class AgentState(TypedDict):
    messages: list  # Conversation history
    
def agent_think(state: AgentState) -> dict:
    """LLM decides what to do."""
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Bind tools to model so it can call them
    model_with_tools = model.bind_tools([search_wikipedia, fetch_url])
    
    # Get latest user message + history
    response = model_with_tools.invoke(state["messages"])
    
    # Add LLM response to messages
    return {"messages": state["messages"] + [response]}

def should_continue(state: AgentState) -> str:
    """Check if LLM wants to use tools or is done."""
    last_message = state["messages"][-1]
    
    # If LLM response has tool calls, continue to tool execution
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "end"  # LLM is done, no more tools needed

def execute_tools(state: AgentState) -> dict:
    """Execute whatever tools LLM requested."""
    last_message = state["messages"][-1]
    
    new_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        
        # Call the tool
        if tool_name == "search_wikipedia":
            result = search_wikipedia.invoke(tool_input)
        elif tool_name == "fetch_url":
            result = fetch_url.invoke(tool_input)
        else:
            result = "Unknown tool"
        
        # Add tool result to messages
        new_messages.append(ToolMessage(
            tool_call_id=tool_call["id"],
            content=result
        ))
    
    return {"messages": state["messages"] + new_messages}

# Build agent graph
builder = StateGraph(AgentState)
builder.add_node("think", agent_think)
builder.add_node("tools", execute_tools)

builder.add_edge(START, "think")
builder.add_conditional_edges(
    "think",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
builder.add_edge("tools", "think")  # Loop back to think

graph = builder.compile()

# Run agent
result = graph.invoke({
    "messages": [HumanMessage(content="Tell me about Python programming")]
})

print(result["messages"][-1].content)
```

---

## Multi-Step Decision Trees

Complex workflows with multiple branches:

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class ProcessState(TypedDict):
    request: str
    priority: str
    assigned_team: str
    status: str

def assess_priority(state: ProcessState) -> dict:
    """Determine if request is high or low priority."""
    priority = "high" if any(word in state["request"].lower() 
                            for word in ["urgent", "critical", "asap"])
    return {"priority": priority}

def assign_high_priority(state: ProcessState) -> dict:
    return {"assigned_team": "Senior Team", "status": "expedited"}

def assign_low_priority(state: ProcessState) -> dict:
    return {"assigned_team": "Junior Team", "status": "standard"}

def process_expedited(state: ProcessState) -> dict:
    return {"status": f"{state['status']} - processing with priority"}

def process_standard(state: ProcessState) -> dict:
    return {"status": f"{state['status']} - processing in queue"}

def router_priority(state: ProcessState) -> Literal["high", "low"]:
    return state["priority"]

def router_team(state: ProcessState) -> Literal["expedited", "standard"]:
    return "expedited" if state["priority"] == "high" else "standard"

builder = StateGraph(ProcessState)

# Nodes
builder.add_node("assess", assess_priority)
builder.add_node("high_team", assign_high_priority)
builder.add_node("low_team", assign_low_priority)
builder.add_node("expedited", process_expedited)
builder.add_node("standard", process_standard)

# Edges
builder.add_edge(START, "assess")

# First branch: priority routing
builder.add_conditional_edges(
    "assess",
    router_priority,
    {
        "high": "high_team",
        "low": "low_team"
    }
)

# Converge back and second branch: processing routing
builder.add_edge("high_team", "expedited")
builder.add_edge("low_team", "standard")

builder.add_edge("expedited", END)
builder.add_edge("standard", END)

graph = builder.compile()

# Test
result = graph.invoke({
    "request": "URGENT: Fix production bug",
    "priority": "", "assigned_team": "", "status": ""
})
print(f"Team: {result['assigned_team']}, Status: {result['status']}")
# Output: Team: Senior Team, Status: expedited - processing with priority
```

---

## State Persistence & Memory

### Problem: Without Memory
Each `graph.invoke()` call starts fresh — no history.

### Solution 1: InMemorySaver (Development)
```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "user-123"}}

# First call
result1 = graph.invoke({"message": "Hello"}, config)

# Second call — same thread_id, so it remembers the first call
result2 = graph.invoke({"message": "How are you?"}, config)

# Each invoke sees full history in state
```

### Solution 2: Stateful Agent (Production)
For production, use a database checkpointer:

```python
# Install: pip install psycopg2-binary
from langgraph.checkpoint.postgres import PostgresSaver

# Connect to database
connection_string = "postgresql://user:password@localhost/langgraph"
checkpointer = PostgresSaver(connection_string)

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({...}, config)  # Persisted to DB
```

---

## Real Example: Loan Eligibility Agent

Complete, production-ready loan workflow:

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import os

os.environ["OPENAI_API_KEY"] = "your-key"

@tool
def get_credit_score(pan: str) -> int:
    """Fetch CIBIL credit score for PAN."""
    fake_bureau = {"ABCDE1234F": 762, "XYZAB9876K": 640}
    return fake_bureau.get(pan, 700)

@tool
def calculate_foir(monthly_income: float, existing_emi: float) -> float:
    """Calculate FOIR (Fixed Obligation to Income Ratio)."""
    return round((existing_emi / monthly_income) * 100, 1)

@tool
def check_employment(pan: str) -> str:
    """Verify employment status."""
    employment = {"ABCDE1234F": "Salaried", "XYZAB9876K": "Self-employed"}
    return employment.get(pan, "Unknown")

# Create agent with tools
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_credit_score, calculate_foir, check_employment],
    system_prompt=(
        "You are a loan eligibility expert. Use tools to evaluate applicants. "
        "Recommend approval only if:\n"
        "- Credit score >= 700\n"
        "- FOIR <= 50%\n"
        "- Employed (any type)\n"
        "Always explain your reasoning."
    ),
    checkpointer=InMemorySaver()
)

# Test
config = {"configurable": {"thread_id": "loan-app-001"}}

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "PAN ABCDE1234F, income 90000, EMI 30000. Eligible?"
        }]
    },
    config
)

print(result["messages"][-1].content)
# Agent will:
# 1. Call get_credit_score("ABCDE1234F") → 762
# 2. Call calculate_foir(90000, 30000) → 33.3
# 3. Call check_employment("ABCDE1234F") → "Salaried"
# 4. Reason through and recommend approval
```

---

## Key Patterns in Intermediate LangGraph

| Pattern | Use Case | Example |
|---|---|---|
| **LLM Chain** | Structured prompts + parsing | Summarization, classification |
| **Tool Calling** | LLM decides when to fetch data | Weather bot, calculator |
| **Agent Loop** | Iterative thinking + action | Research bot, troubleshooting |
| **Multi-branch** | Complex decision trees | Routing, triage systems |
| **Memory/State** | Preserve conversation history | Chatbots, multi-turn workflows |

---

## Common Issues & Solutions

### Issue 1: Tool Not Getting Called
**Problem:** LLM doesn't use tools even though defined.
**Solution:** Make tool descriptions clear and use system prompt to encourage tool use.

```python
@tool
def get_data(query: str) -> str:
    """Search and fetch relevant information. Use this for any factual questions."""
    # Better description = more likely LLM uses it
```

### Issue 2: Infinite Loop
**Problem:** Agent keeps looping, never terminates.
**Solution:** Define clear end conditions in router.

```python
def should_continue(state):
    # Make sure at least one path leads to END
    if some_condition:
        return "end"  # Must go to END eventually
```

### Issue 3: State Not Persisting
**Problem:** Second invoke() call doesn't remember first call.
**Solution:** Use checkpointer and same thread_id.

```python
config = {"configurable": {"thread_id": "same-id"}}
graph.invoke({...}, config)  # Call 1
graph.invoke({...}, config)  # Call 2 — remembers call 1
```

---

## Next Steps

You now understand:
- ✅ LLM integration
- ✅ Tools and tool calling
- ✅ Agent loops
- ✅ Multi-step workflows
- ✅ State persistence

**Ready for Advanced?** Move to `03_LANGGRAPH_ADVANCED.md` to learn:
- Streaming and real-time updates
- Complex subgraphs
- Error handling and retries
- Performance optimization
- Production deployment patterns
