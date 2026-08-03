# LangGraph Cheat Sheet

A quick reference for common patterns and syntax.

---

## Setup

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

---

## Define State

```python
class MyState(TypedDict):
    user_input: str
    response: str
    count: int
    metadata: dict
```

---

## Create Nodes

### Simple Node
```python
def my_node(state: MyState) -> dict:
    return {"response": "done"}
```

### LLM Node
```python
def llm_node(state: MyState) -> dict:
    model = ChatOpenAI(model="gpt-4o-mini")
    result = model.invoke(state["user_input"])
    return {"response": result.content}
```

### LLM with Chain
```python
def chain_node(state: MyState) -> dict:
    model = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("Summarize: {text}")
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"text": state["user_input"]})
    return {"response": result}
```

### Tool Node
```python
@tool
def my_tool(query: str) -> str:
    """Tool description for LLM."""
    return f"Result: {query}"
```

---

## Build Graph

### Basic Structure
```python
builder = StateGraph(MyState)

# Add nodes
builder.add_node("node_name", node_function)

# Add edges
builder.add_edge(START, "node_name")
builder.add_edge("node_name", END)

# Compile
graph = builder.compile()
```

### Linear Workflow
```python
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", "step3")
builder.add_edge("step3", END)
```

### Conditional Routing
```python
def router(state: MyState) -> Literal["path_a", "path_b"]:
    return "path_a" if condition else "path_b"

builder.add_conditional_edges(
    "decision_node",
    router,
    {
        "path_a": "node_a",
        "path_b": "node_b"
    }
)
```

### Add Memory (Checkpointer)
```python
graph = builder.compile(checkpointer=InMemorySaver())
```

---

## Run Graph

### Basic Invoke
```python
result = graph.invoke({"user_input": "Hello", "response": "", "count": 0, "metadata": {}})
print(result["response"])
```

### With Thread ID (Persistent Memory)
```python
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"user_input": "Hello", ...}, config)
```

### Streaming
```python
async def stream():
    async for event in graph.astream_events(
        {"user_input": "Hello", ...},
        version="v2"
    ):
        if event["event"] == "on_chain_stream":
            print(event["data"]["chunk"]["content"], end="", flush=True)

import asyncio
asyncio.run(stream())
```

---

## Error Handling

### Try-Except in Node
```python
def safe_node(state: MyState) -> dict:
    try:
        result = risky_operation()
        return {"response": result}
    except Exception as e:
        return {"response": f"Error: {e}"}
```

### Retry Pattern
```python
import time

def node_with_retry(state: MyState, max_retries=3) -> dict:
    for attempt in range(max_retries):
        try:
            return {"response": do_work()}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"response": f"Failed: {e}"}
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

## Common Patterns

### Parallel Execution
```python
builder.add_edge(START, "common_node")
builder.add_edge("common_node", "task_a")
builder.add_edge("common_node", "task_b")
builder.add_edge("task_a", END)
builder.add_edge("task_b", END)
```

### Loops
```python
def should_continue(state) -> Literal["continue", "stop"]:
    return "continue" if state["count"] < 10 else "stop"

builder.add_conditional_edges(
    "process",
    should_continue,
    {"continue": "process", "stop": END}  # Loop back to self
)
```

### Merge Multiple Paths
```python
builder.add_edge("path_a", "merge_node")
builder.add_edge("path_b", "merge_node")
builder.add_edge("merge_node", END)
```

---

## State Updates

### Partial Dict Merge
```python
# Node returns only fields to update
def node(state: MyState) -> dict:
    return {"response": "new_value"}  # count stays same

# State after node executes:
# {"user_input": "...", "response": "new_value", "count": 0}
```

### Update Multiple Fields
```python
def node(state: MyState) -> dict:
    return {
        "response": "new",
        "count": state["count"] + 1,
        "metadata": {"status": "done"}
    }
```

---

## LLM Integration

### Basic LLM Call
```python
model = ChatOpenAI(api_key="...", model="gpt-4o-mini", temperature=0.7)
response = model.invoke("Hello")
print(response.content)
```

### With Tools (Auto Tool Calling)
```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[my_tool1, my_tool2],
    system_prompt="You are helpful.",
    checkpointer=InMemorySaver()
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Question?"}]},
    {"configurable": {"thread_id": "1"}}
)
print(result["messages"][-1].content)
```

### Structured Output
```python
from typing import TypedDict

class OutputFormat(TypedDict):
    name: str
    age: int

model = ChatOpenAI(model="gpt-4o-mini")
parser = model.with_structured_output(OutputFormat)
result = parser.invoke("Extract name and age from text")
print(result["name"], result["age"])
```

---

## Type Hints (TypedDict)

```python
from typing import TypedDict, Optional, List

class FullState(TypedDict):
    required_field: str           # Required
    number: int
    optional_field: Optional[str] # Can be None
    items: List[str]             # List of strings
    nested: dict                 # Dict (any structure)
```

---

## Router Function

### Single Condition
```python
def simple_router(state: MyState) -> Literal["path_a", "path_b"]:
    return "path_a" if condition else "path_b"
```

### Multiple Conditions
```python
def complex_router(state: MyState) -> Literal["urgent", "normal", "low"]:
    if state["priority"] == "high":
        return "urgent"
    elif state["priority"] == "medium":
        return "normal"
    else:
        return "low"
```

---

## Production Deployment

### Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

### FastAPI Wrapper
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    prompt: str

@app.post("/invoke")
def invoke(input: Input):
    result = graph.invoke({"user_input": input.prompt, ...})
    return {"response": result["response"]}
```

### Environment Variables
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

---

## Debugging Tips

### Print State
```python
def debug_node(state: MyState) -> dict:
    print(f"DEBUG: {state}")
    return {}
```

### Log Node Execution
```python
import logging
logger = logging.getLogger(__name__)

def logged_node(state: MyState) -> dict:
    logger.info(f"Starting node, input={state['user_input']}")
    result = do_work()
    logger.info(f"Completed, output={result}")
    return {"response": result}
```

### Visualize Graph
```python
# LangGraph can generate a visualization
try:
    image = graph.get_graph().draw_mermaid_png()
    # Save or display the image
except Exception as e:
    print(f"Visualization not available: {e}")
```

---

## Common Mistakes

| Mistake | Fix |
|---|---|
| Router returns wrong node name | Make sure return value matches a defined edge |
| Forgot END edge | Every path must lead to `END` |
| Not initializing all state fields | Provide all TypedDict fields in `invoke()` |
| State not persisting | Use `checkpointer` and same `thread_id` |
| LLM not using tools | Write clear tool descriptions, encourage in prompt |
| Graph loops forever | Ensure conditional edges have escape condition |

---

## One-Liner Reference

```python
# Create and run a simple graph in 10 lines
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    x: str

def node(state): return {"x": state["x"].upper()}

b = StateGraph(S)
b.add_node("go", node)
b.add_edge(START, "go")
b.add_edge("go", END)

result = b.compile().invoke({"x": "hello"})
print(result["x"])  # "HELLO"
```

---

## Performance Tips

1. **Cache expensive operations** → Use `@lru_cache`
2. **Batch process** → Process multiple items in one node
3. **Run independent tasks in parallel** → Use multiple edges from same node
4. **Use streaming** → Send results to user as they arrive
5. **Optimize prompts** → Shorter/clearer prompts = faster LLM calls

---

## When to Use What

| Need | Use |
|---|---|
| Simple sequence | Linear edges: `add_edge(A, B)` |
| Conditional path | Router + `add_conditional_edges()` |
| LLM decision | Put LLM in node, use router |
| Parallel work | Multiple `add_edge()` from same node |
| Tool calling | Use `create_agent()` or bind tools to model |
| Remember state | Add `checkpointer=InMemorySaver()` |
| Stream results | Use `astream_events()` |

---

## Key Functions

```python
StateGraph()                          # Create graph
.add_node(name, func)                # Add a node
.add_edge(from, to)                  # Add linear edge
.add_conditional_edges(from, router) # Add branching
.compile(checkpointer=...)           # Finalize
.invoke(state, config)               # Run once
.astream_events(state, version="v2") # Stream async
```

---

**Keep this sheet nearby while coding!** 🚀
