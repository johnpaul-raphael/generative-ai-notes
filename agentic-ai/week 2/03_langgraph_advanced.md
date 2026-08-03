# LangGraph for Advanced: Production Workflows & Optimization

## Table of Contents
1. [Streaming & Real-Time Updates](#streaming--real-time-updates)
2. [Subgraphs & Composition](#subgraphs--composition)
3. [Error Handling & Retries](#error-handling--retries)
4. [Performance Optimization](#performance-optimization)
5. [Production Deployment](#production-deployment)
6. [Advanced Patterns](#advanced-patterns)

---

## Streaming & Real-Time Updates

### Why Streaming?
Traditional `graph.invoke()` blocks until done. For long-running LLM tasks, stream intermediate results to user in real-time.

### Streaming with astream_events()

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio

class StreamState(TypedDict):
    topic: str
    essay: str

def write_essay(state: StreamState) -> dict:
    """LLM writes an essay (streaming)."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "Write a 3-paragraph essay on: {topic}"
    )
    chain = prompt | model | StrOutputParser()
    
    # Note: StrOutputParser doesn't support streaming in some contexts
    # For true streaming, use model.stream() directly
    result = chain.invoke({"topic": state["topic"]})
    return {"essay": result}

builder = StateGraph(StreamState)
builder.add_node("write", write_essay)
builder.add_edge(START, "write")
builder.add_edge("write", END)

graph = builder.compile()

# Streaming approach 1: Stream chunks as they arrive
async def stream_results():
    async for event in graph.astream_events(
        {"topic": "Artificial Intelligence", "essay": ""},
        config={"configurable": {"thread_id": "stream-1"}},
        version="v2"
    ):
        if event["event"] == "on_chain_stream":
            print(event["data"]["chunk"]["content"], end="", flush=True)

# Run streaming
asyncio.run(stream_results())
```

### Streaming Approach 2: Token-by-Token LLM Streaming

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

# Streaming directly from model
for chunk in model.stream("Tell me a joke"):
    print(chunk.content, end="", flush=True)
print()
```

### Streaming in Web Apps (FastAPI Example)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()
graph = builder.compile()  # Your compiled graph

@app.post("/ask")
async def ask_agent(query: str):
    async def generate():
        async for event in graph.astream_events(
            {"topic": query, "essay": ""},
            version="v2"
        ):
            if event["event"] == "on_chain_stream":
                chunk = event["data"]["chunk"]["content"]
                yield f"data: {chunk}\n\n"  # SSE format for client
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Subgraphs & Composition

### Problem: Large Graphs Get Messy
When a graph has 20+ nodes, it becomes hard to manage. **Subgraphs** let you modularize workflows.

### Concept: Graph Composition

A subgraph is a graph used as a node inside another graph.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ============= SUBGRAPH 1: Validation =============
class ValidateState(TypedDict):
    data: str
    is_valid: bool
    error: str

def validate_format(state: ValidateState) -> dict:
    is_valid = len(state["data"]) > 0
    error = "" if is_valid else "Data is empty"
    return {"is_valid": is_valid, "error": error}

validate_builder = StateGraph(ValidateState)
validate_builder.add_node("validate", validate_format)
validate_builder.add_edge(START, "validate")
validate_builder.add_edge("validate", END)
validate_graph = validate_builder.compile()

# ============= SUBGRAPH 2: Processing =============
class ProcessState(TypedDict):
    data: str
    processed: str

def process_data(state: ProcessState) -> dict:
    processed = state["data"].upper() + "!"
    return {"processed": processed}

process_builder = StateGraph(ProcessState)
process_builder.add_node("process", process_data)
process_builder.add_edge(START, "process")
process_builder.add_edge("process", END)
process_graph = process_builder.compile()

# ============= MAIN GRAPH: Orchestration =============
class MainState(TypedDict):
    data: str
    is_valid: bool
    error: str
    processed: str

def run_validate(state: MainState) -> dict:
    """Run validation subgraph."""
    result = validate_graph.invoke({
        "data": state["data"],
        "is_valid": False,
        "error": ""
    })
    return {
        "is_valid": result["is_valid"],
        "error": result["error"]
    }

def run_process(state: MainState) -> dict:
    """Run processing subgraph."""
    result = process_graph.invoke({
        "data": state["data"],
        "processed": ""
    })
    return {"processed": result["processed"]}

main_builder = StateGraph(MainState)
main_builder.add_node("validate", run_validate)
main_builder.add_node("process", run_process)

main_builder.add_edge(START, "validate")
main_builder.add_edge("validate", "process")
main_builder.add_edge("process", END)

main_graph = main_builder.compile()

# Run main graph
result = main_graph.invoke({
    "data": "hello",
    "is_valid": False,
    "error": "",
    "processed": ""
})
print(result)
# Output: {'data': 'hello', 'is_valid': True, 'error': '', 'processed': 'HELLO!'}
```

### Using add_edge with Subgraphs

For advanced composition, LangGraph allows **wrapping** subgraph results:

```python
# Use nodebuilder.add_node("subgraph_name", subgraph_instance) 
# where subgraph_instance is a compiled graph
builder.add_node("validate", validate_graph)  # Directly use compiled graph
```

---

## Error Handling & Retries

### Try-Except in Nodes

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

class SafeState(TypedDict):
    prompt: str
    response: str
    error: str

def safe_llm_call(state: SafeState) -> dict:
    """Call LLM with error handling."""
    try:
        model = ChatOpenAI(model="gpt-4o-mini")
        result = model.invoke(state["prompt"])
        return {"response": result.content, "error": ""}
    
    except Exception as e:
        # Catch API errors, network errors, etc.
        return {"response": "", "error": f"LLM error: {str(e)}"}

builder = StateGraph(SafeState)
builder.add_node("safe_llm", safe_llm_call)
builder.add_edge(START, "safe_llm")
builder.add_edge("safe_llm", END)

graph = builder.compile()
result = graph.invoke({"prompt": "Hello", "response": "", "error": ""})
print(f"Response: {result['response']}")
print(f"Error: {result['error']}")
```

### Retry Pattern with Exponential Backoff

```python
import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class RetryState(TypedDict):
    task: str
    attempts: int
    max_attempts: int
    result: str
    success: bool

def unreliable_task(state: RetryState) -> dict:
    """A task that might fail."""
    import random
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("Random failure")
    return {"result": "Success!", "success": True}

def retry_with_backoff(state: RetryState) -> dict:
    """Execute task with retries."""
    for attempt in range(state["max_attempts"]):
        try:
            result = unreliable_task(state)
            return {**result, "attempts": attempt + 1}
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    return {
        "result": "Failed after max attempts",
        "success": False,
        "attempts": state["max_attempts"]
    }

builder = StateGraph(RetryState)
builder.add_node("retry_task", retry_with_backoff)
builder.add_edge(START, "retry_task")
builder.add_edge("retry_task", END)

graph = builder.compile()
result = graph.invoke({
    "task": "risky_operation",
    "attempts": 0,
    "max_attempts": 3,
    "result": "",
    "success": False
})
print(f"Success: {result['success']}, Attempts: {result['attempts']}")
```

### Fallback Chains

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class FallbackState(TypedDict):
    query: str
    result: str
    source: str

def primary_source(state: FallbackState) -> dict:
    """Try to get data from primary source."""
    try:
        # Simulated primary API
        if state["query"] == "known":
            return {"result": "Data from primary", "source": "primary"}
        raise Exception("Not found in primary")
    except Exception:
        return None  # Signal failure

def fallback_source(state: FallbackState) -> dict:
    """Use fallback if primary fails."""
    return {"result": "Data from fallback", "source": "fallback"}

def try_primary_then_fallback(state: FallbackState) -> dict:
    """Try primary, fall back if needed."""
    primary = primary_source(state)
    if primary:
        return primary
    return fallback_source(state)

builder = StateGraph(FallbackState)
builder.add_node("fetch", try_primary_then_fallback)
builder.add_edge(START, "fetch")
builder.add_edge("fetch", END)

graph = builder.compile()
result = graph.invoke({"query": "test", "result": "", "source": ""})
print(result)  # {'query': 'test', 'result': 'Data from fallback', 'source': 'fallback'}
```

---

## Performance Optimization

### 1. Parallel Node Execution

Instead of A → B → C sequentially, run B and C in parallel after A.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import time

class ParallelState(TypedDict):
    input: str
    result_b: str
    result_c: str

def node_a(state: ParallelState) -> dict:
    return {"input": state["input"] + "_processed"}

def node_b(state: ParallelState) -> dict:
    time.sleep(2)  # Simulate long task
    return {"result_b": f"B: {state['input']}"}

def node_c(state: ParallelState) -> dict:
    time.sleep(2)  # Simulate long task
    return {"result_c": f"C: {state['input']}"}

builder = StateGraph(ParallelState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)

builder.add_edge(START, "a")
# Both B and C start after A completes (parallel execution in LangGraph)
builder.add_edge("a", "b")
builder.add_edge("a", "c")

# Merge point: both must finish before next node
builder.add_edge("b", END)
builder.add_edge("c", END)

graph = builder.compile()

start = time.time()
result = graph.invoke({"input": "data", "result_b": "", "result_c": ""})
elapsed = time.time() - start

print(f"Completed in {elapsed:.1f}s")
# Note: Sequential would take ~4s (2+2), parallel should be ~2s
```

### 2. Caching & Memoization

```python
from functools import lru_cache
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

@lru_cache(maxsize=128)
def expensive_computation(key: str) -> str:
    """Cache results of expensive calls."""
    print(f"Computing {key}...")
    time.sleep(1)
    return f"Result_{key}"

class CacheState(TypedDict):
    query: str
    result: str

def cached_node(state: CacheState) -> dict:
    result = expensive_computation(state["query"])
    return {"result": result}

builder = StateGraph(CacheState)
builder.add_node("compute", cached_node)
builder.add_edge(START, "compute")
builder.add_edge("compute", END)

graph = builder.compile()

# First call: computes
result1 = graph.invoke({"query": "test", "result": ""})
# Second call with same query: uses cache (instant)
result2 = graph.invoke({"query": "test", "result": ""})
# Different query: computes again
result3 = graph.invoke({"query": "other", "result": ""})
```

### 3. Batch Processing

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class BatchState(TypedDict):
    items: list
    processed: list

def process_batch(state: BatchState) -> dict:
    """Process multiple items at once."""
    # More efficient than one-by-one
    processed = [item.upper() for item in state["items"]]
    return {"processed": processed}

builder = StateGraph(BatchState)
builder.add_node("batch", process_batch)
builder.add_edge(START, "batch")
builder.add_edge("batch", END)

graph = builder.compile()

result = graph.invoke({"items": ["a", "b", "c"], "processed": []})
print(result["processed"])  # ['A', 'B', 'C']
```

---

## Production Deployment

### 1. Containerization with Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. FastAPI Wrapper

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

app = FastAPI()

# Your graph (build as shown earlier)
builder = StateGraph(YourState)
# ... add nodes, edges ...
graph = builder.compile(checkpointer=PostgresSaver(...))

class InputRequest(BaseModel):
    prompt: str
    thread_id: str

@app.post("/invoke")
def invoke_graph(req: InputRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke(
        {"prompt": req.prompt, ...},
        config
    )
    return {"result": result}

@app.get("/health")
def health():
    return {"status": "ok"}
```

### 3. Environment Variables & Secrets

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env

api_key = os.getenv("OPENAI_API_KEY")
db_url = os.getenv("DATABASE_URL")

if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
```

### 4. Logging & Monitoring

```python
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoggedState(TypedDict):
    event: str
    status: str

def logged_node(state: LoggedState) -> dict:
    logger.info(f"Processing event: {state['event']}")
    try:
        # Do work
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing {state['event']}: {e}")
        return {"status": "failed"}

builder = StateGraph(LoggedState)
builder.add_node("work", logged_node)
builder.add_edge(START, "work")
builder.add_edge("work", END)

graph = builder.compile()
```

---

## Advanced Patterns

### Pattern 1: Dynamic Node Creation

```python
class DynamicState(TypedDict):
    tasks: list
    completed: list

def create_nodes_dynamically(tasks):
    """Create nodes based on tasks."""
    builder = StateGraph(DynamicState)
    
    for task in tasks:
        def make_node(t):
            def execute(state):
                return {"completed": state["completed"] + [f"{t}_done"]}
            return execute
        
        builder.add_node(f"task_{task}", make_node(task))
    
    # Wire them up
    builder.add_edge(START, f"task_{tasks[0]}")
    for i in range(len(tasks) - 1):
        builder.add_edge(f"task_{tasks[i]}", f"task_{tasks[i+1]}")
    builder.add_edge(f"task_{tasks[-1]}", END)
    
    return builder.compile()

graph = create_nodes_dynamically(["a", "b", "c"])
result = graph.invoke({"tasks": ["a", "b", "c"], "completed": []})
print(result["completed"])
```

### Pattern 2: Graph Feedback Loops

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class LoopState(TypedDict):
    iteration: int
    max_iterations: int
    value: int

def increment(state: LoopState) -> dict:
    return {"value": state["value"] + 1, "iteration": state["iteration"] + 1}

def should_loop(state: LoopState) -> Literal["continue", "done"]:
    return "continue" if state["iteration"] < state["max_iterations"] else "done"

builder = StateGraph(LoopState)
builder.add_node("increment", increment)

builder.add_edge(START, "increment")
builder.add_conditional_edges(
    "increment",
    should_loop,
    {
        "continue": "increment",  # Loop back
        "done": END
    }
)

graph = builder.compile()
result = graph.invoke({"iteration": 0, "max_iterations": 5, "value": 0})
print(result["value"])  # 5 (after 5 iterations)
```

### Pattern 3: Human-in-the-Loop

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class HumanLoopState(TypedDict):
    decision: str
    approved: bool
    result: str

def generate_recommendation(state: HumanLoopState) -> dict:
    return {"decision": "Approve loan", "approved": False}

def wait_for_human(state: HumanLoopState) -> dict:
    # In production, this would halt and wait for human input
    # For now, simulate human approval
    human_approval = input("Approve? (yes/no): ") == "yes"
    return {"approved": human_approval}

def process_if_approved(state: HumanLoopState) -> dict:
    if state["approved"]:
        return {"result": "Loan processed"}
    else:
        return {"result": "Loan rejected"}

def should_proceed(state: HumanLoopState) -> Literal["process", "reject"]:
    return "process" if state["approved"] else "reject"

builder = StateGraph(HumanLoopState)
builder.add_node("generate", generate_recommendation)
builder.add_node("human", wait_for_human)
builder.add_node("process", process_if_approved)

builder.add_edge(START, "generate")
builder.add_edge("generate", "human")
builder.add_conditional_edges(
    "human",
    should_proceed,
    {"process": "process", "reject": "process"}
)
builder.add_edge("process", END)

graph = builder.compile()
```

---

## Performance Benchmarking

```python
import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class BenchState(TypedDict):
    data: str

def benchmark_graph(graph, num_runs=100):
    """Measure graph execution time."""
    times = []
    for _ in range(num_runs):
        start = time.time()
        graph.invoke({"data": "test"})
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average: {avg_time:.3f}s, Min: {min(times):.3f}s, Max: {max(times):.3f}s")

benchmark_graph(graph, num_runs=100)
```

---

## Summary: When to Use Advanced Patterns

| Pattern | When to Use |
|---|---|
| **Streaming** | Long-running tasks, real-time user feedback |
| **Subgraphs** | Large workflows, code reuse |
| **Error Handling** | External APIs, unreliable sources |
| **Retries** | Network-dependent tasks |
| **Parallel Execution** | Independent tasks that can run concurrently |
| **Caching** | Expensive computations with repeated inputs |
| **Human-in-Loop** | Decisions requiring human review |
| **Feedback Loops** | Iterative refinement, self-healing |

---

## Deployment Checklist

- [ ] Tests written and passing
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Environment variables set
- [ ] Database (checkpointer) configured
- [ ] API endpoints defined
- [ ] Rate limiting implemented
- [ ] Monitoring/alerting set up
- [ ] Docker image built and tested
- [ ] Documentation complete

---

## Next Steps

You've now learned LangGraph from fundamentals to production. Next:
1. **Build a real project** using these patterns
2. **Contribute to LangGraph** — it's open source
3. **Explore LangServe** — API serving for LangGraph apps
4. **Study advanced use cases** in LangChain docs

Good luck building powerful agentic applications!
