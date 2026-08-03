# LangGraph for Beginners: Build Your First Agent Workflow

## Table of Contents
1. [What is LangGraph?](#what-is-langgraph)
2. [Core Concepts](#core-concepts)
3. [Your First Graph (Hello World)](#your-first-graph)
4. [Understanding State](#understanding-state)
5. [Linear Workflows](#linear-workflows)
6. [Conditional Routing](#conditional-routing)
7. [Complete Example: Simple Chatbot](#complete-example)

---

## What is LangGraph?

LangGraph is a **framework for building stateful, multi-step AI applications** where:
- Each step is a node (a Python function or LLM call)
- Data flows between steps through state
- Steps can branch or loop based on conditions
- Everything is a directed graph (like a flowchart)

**Real-world analogy:** Think of LangGraph as building a process with decision points:
```
Start → Step 1 → Step 2 → Decision → Step 3A OR Step 3B → End
```

### Why LangGraph (not just LangChain)?
- **LangChain**: Simple linear chains (A → B → C)
- **LangGraph**: Complex workflows with branches, loops, and memory

| Feature | LangChain | LangGraph |
|---|---|---|
| Simple sequences | ✅ | ✅ |
| Branches/routing | ❌ | ✅ |
| Loops | ❌ | ✅ |
| State management | Limited | ✅ |
| Memory persistence | ❌ | ✅ |

---

## Core Concepts

### 1. **State** — The Data Container
State is a `TypedDict` (a dictionary with type hints) that flows through your graph.

```python
from typing import TypedDict

class MyState(TypedDict):
    """Data that moves through the graph."""
    user_input: str      # What the user said
    response: str        # What we calculated
    count: int          # A counter
```

### 2. **Nodes** — The Workers
Nodes are Python functions that:
- Take state as input
- Do some work
- Return a partial dict to update state

```python
def process_input(state: MyState) -> dict:
    """A node that processes input."""
    # Read from state
    user_input = state["user_input"]
    
    # Do work
    response = f"You said: {user_input}"
    
    # Return partial dict to update state
    return {"response": response}
```

### 3. **Edges** — The Connections
Edges connect nodes (like arrows in a flowchart).

```python
builder.add_edge("node_a", "node_b")  # Always go from A to B
```

### 4. **Graph** — The Container
StateGraph holds all nodes and edges together.

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(MyState)
builder.add_node("my_node", process_input)
builder.add_edge(START, "my_node")  # START is special — entry point
builder.add_edge("my_node", END)    # END is special — exit point
graph = builder.compile()
```

### 5. **Invoke** — The Execution
Run your graph with initial state.

```python
result = graph.invoke({"user_input": "Hello"})
print(result["response"])  # "You said: Hello"
```

---

## Your First Graph (Hello World)

### Step 1: Import Required Libraries
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Install: pip install langgraph
```

### Step 2: Define State
```python
class SimpleState(TypedDict):
    message: str
    processed: str
```

### Step 3: Create a Node
```python
def greet(state: SimpleState) -> dict:
    """A simple node that greets."""
    greeting = f"Hello, {state['message']}!"
    return {"processed": greeting}
```

### Step 4: Build the Graph
```python
builder = StateGraph(SimpleState)
builder.add_node("greet", greet)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)
```

### Step 5: Compile and Run
```python
graph = builder.compile()

result = graph.invoke({
    "message": "Alice",
    "processed": ""  # Required by TypedDict, but we'll update it
})

print(result["processed"])  # Output: "Hello, Alice!"
```

### Full Code
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class SimpleState(TypedDict):
    message: str
    processed: str

def greet(state: SimpleState) -> dict:
    greeting = f"Hello, {state['message']}!"
    return {"processed": greeting}

builder = StateGraph(SimpleState)
builder.add_node("greet", greet)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)

graph = builder.compile()
result = graph.invoke({"message": "Alice", "processed": ""})
print(result["processed"])
```

**Output:**
```
Hello, Alice!
```

---

## Understanding State

### What is State?
State is the **shared memory** between nodes. Each node reads from it and updates it.

### Rules for State Updates
1. **Nodes return a partial dict** — only fields they want to update
2. **State merges the updates** — old values + new values = final state
3. **Each node sees the full merged state** — including updates from previous nodes

### Example: Multi-Step State Updates
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class CountState(TypedDict):
    count: int
    log: str

def increment(state: CountState) -> dict:
    """Increment count."""
    new_count = state["count"] + 1
    new_log = state["log"] + "incremented, "
    return {"count": new_count, "log": new_log}

def double(state: CountState) -> dict:
    """Double count."""
    new_count = state["count"] * 2
    new_log = state["log"] + "doubled, "
    return {"count": new_count, "log": new_log}

def square(state: CountState) -> dict:
    """Square count."""
    new_count = state["count"] ** 2
    new_log = state["log"] + "squared"
    return {"count": new_count, "log": new_log}

builder = StateGraph(CountState)
builder.add_node("increment", increment)
builder.add_node("double", double)
builder.add_node("square", square)

builder.add_edge(START, "increment")
builder.add_edge("increment", "double")
builder.add_edge("double", "square")
builder.add_edge("square", END)

graph = builder.compile()

result = graph.invoke({"count": 2, "log": ""})
print(f"Final count: {result['count']}")      # (2+1) * 2 = 6, 6^2 = 36
print(f"Operations: {result['log']}")  # "incremented, doubled, squared"
```

**Output:**
```
Final count: 36
Operations: incremented, doubled, squared
```

---

## Linear Workflows

A **linear workflow** is nodes connected in a straight line: A → B → C → D

### Use Case: Text Processing Pipeline
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class TextState(TypedDict):
    raw_text: str
    cleaned: str
    uppercase: str
    final: str

def clean_text(state: TextState) -> dict:
    """Remove extra spaces."""
    cleaned = " ".join(state["raw_text"].split())
    return {"cleaned": cleaned}

def to_uppercase(state: TextState) -> dict:
    """Convert to uppercase."""
    uppercase = state["cleaned"].upper()
    return {"uppercase": uppercase}

def add_punctuation(state: TextState) -> dict:
    """Add punctuation."""
    final = state["uppercase"] + "!"
    return {"final": final}

# Build graph
builder = StateGraph(TextState)
builder.add_node("clean", clean_text)
builder.add_node("uppercase", to_uppercase)
builder.add_node("punctuate", add_punctuation)

builder.add_edge(START, "clean")
builder.add_edge("clean", "uppercase")
builder.add_edge("uppercase", "punctuate")
builder.add_edge("punctuate", END)

graph = builder.compile()

result = graph.invoke({
    "raw_text": "hello    world",
    "cleaned": "",
    "uppercase": "",
    "final": ""
})

print(result["final"])  # "HELLO WORLD!"
```

---

## Conditional Routing

**Conditional routing** means: based on state, go to node A or node B (not both).

### Concept: The Router Function
A router is a function that inspects state and returns which node to go to next.

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class NumberState(TypedDict):
    number: int
    result: str

def check_number(state: NumberState) -> dict:
    """Store whether number is even or odd."""
    is_even = state["number"] % 2 == 0
    return {"result": "even" if is_even else "odd"}

def handle_even(state: NumberState) -> dict:
    """Process even numbers."""
    return {"result": f"Number {state['number']} is even"}

def handle_odd(state: NumberState) -> dict:
    """Process odd numbers."""
    return {"result": f"Number {state['number']} is odd"}

# Router function: returns the node name to go to
def router(state: NumberState) -> Literal["even_path", "odd_path"]:
    """Route based on the initial result."""
    if state["result"] == "even":
        return "even_path"
    else:
        return "odd_path"

# Build graph
builder = StateGraph(NumberState)
builder.add_node("check", check_number)
builder.add_node("even_path", handle_even)
builder.add_node("odd_path", handle_odd)

builder.add_edge(START, "check")
# Conditional edge: run router, then go to returned node
builder.add_conditional_edges(
    "check",                    # From this node
    router,                     # Run this router function
    {
        "even_path": "even_path",   # If router returns "even_path", go to "even_path" node
        "odd_path": "odd_path"      # If router returns "odd_path", go to "odd_path" node
    }
)
builder.add_edge("even_path", END)
builder.add_edge("odd_path", END)

graph = builder.compile()

# Test with even number
result = graph.invoke({"number": 4, "result": ""})
print(result["result"])  # "Number 4 is even"

# Test with odd number
result = graph.invoke({"number": 5, "result": ""})
print(result["result"])  # "Number 5 is odd"
```

**Graph Flow:**
```
START
  ↓
[check_number]  (classify as even/odd)
  ↓
  ├─ "even" → [even_path] → END
  │
  └─ "odd" → [odd_path] → END
```

---

## Complete Example: Simple Chatbot

Let's build a beginner-friendly chatbot that routes based on intent.

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class ChatState(TypedDict):
    user_message: str
    detected_intent: str
    response: str

def detect_intent(state: ChatState) -> dict:
    """Detect what the user wants."""
    message = state["user_message"].lower()
    
    if any(word in message for word in ["hello", "hi", "hey"]):
        intent = "greeting"
    elif any(word in message for word in ["weather", "rain", "sunny"]):
        intent = "weather"
    elif any(word in message for word in ["bye", "goodbye", "see you"]):
        intent = "goodbye"
    else:
        intent = "unknown"
    
    return {"detected_intent": intent}

def handle_greeting(state: ChatState) -> dict:
    return {"response": "Hello! How can I help you?"}

def handle_weather(state: ChatState) -> dict:
    return {"response": "The weather is sunny today!"}

def handle_goodbye(state: ChatState) -> dict:
    return {"response": "Goodbye! Have a great day!"}

def handle_unknown(state: ChatState) -> dict:
    return {"response": "I'm not sure what you mean. Try greeting, weather, or goodbye."}

def route_intent(state: ChatState) -> Literal["greeting", "weather", "goodbye", "unknown"]:
    return state["detected_intent"]

# Build graph
builder = StateGraph(ChatState)
builder.add_node("detect", detect_intent)
builder.add_node("greeting", handle_greeting)
builder.add_node("weather", handle_weather)
builder.add_node("goodbye", handle_goodbye)
builder.add_node("unknown", handle_unknown)

builder.add_edge(START, "detect")
builder.add_conditional_edges(
    "detect",
    route_intent,
    {
        "greeting": "greeting",
        "weather": "weather",
        "goodbye": "goodbye",
        "unknown": "unknown"
    }
)
builder.add_edge("greeting", END)
builder.add_edge("weather", END)
builder.add_edge("goodbye", END)
builder.add_edge("unknown", END)

graph = builder.compile()

# Test
for msg in ["Hello!", "What's the weather?", "Goodbye", "Random text"]:
    result = graph.invoke({"user_message": msg, "detected_intent": "", "response": ""})
    print(f"User: {msg}")
    print(f"Bot: {result['response']}\n")
```

**Output:**
```
User: Hello!
Bot: Hello! How can I help you?

User: What's the weather?
Bot: The weather is sunny today!

User: Goodbye
Bot: Goodbye! Have a great day!

User: Random text
Bot: I'm not sure what you mean. Try greeting, weather, or goodbye.
```

---

## Next Steps

You now understand:
- ✅ What LangGraph is
- ✅ State, nodes, and edges
- ✅ Linear workflows
- ✅ Conditional routing

**Ready for Intermediate?** Move to `02_LANGGRAPH_INTERMEDIATE.md` to learn:
- LLM integration (ChatOpenAI)
- Tools and tool calling
- Multiple conditional branches
- Error handling

---

## Common Mistakes Beginners Make

### ❌ Mistake 1: Forgetting END edges
```python
# WRONG - graph never terminates
builder.add_edge("my_node", "other_node")

# RIGHT - end the graph
builder.add_edge("my_node", END)
```

### ❌ Mistake 2: Router returns wrong node name
```python
# WRONG - router returns "a", but edge expects "node_a"
def router(state):
    return "a"  # But edges define {"a": "node_a"}

# RIGHT - return exact node names
def router(state):
    return "node_a"
```

### ❌ Mistake 3: Not initializing all state fields
```python
# WRONG - missing "response" field
result = graph.invoke({"user_input": "hello"})

# RIGHT - provide all TypedDict fields
result = graph.invoke({"user_input": "hello", "response": ""})
```

---

## Glossary

| Term | Meaning |
|---|---|
| **State** | Shared dictionary flowing through the graph |
| **Node** | Function that reads state, does work, returns partial dict |
| **Edge** | Connection from one node to another |
| **Router** | Function that decides which node to go to next |
| **START** | Special node representing graph entry point |
| **END** | Special node representing graph exit point |
| **Conditional edge** | Edge that branches based on router output |
| **Graph** | Container of all nodes and edges; compiled to run |
