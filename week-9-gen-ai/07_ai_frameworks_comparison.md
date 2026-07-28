# 🤖 AI Frameworks Comparison Guide
## LangChain, LangGraph & Competitors

---

## Overview

The AI framework ecosystem has three main categories:

| Category | Frameworks | Best For |
|---|---|---|
| **Orchestration** | LangChain, LlamaIndex, Haystack | Building pipelines that connect LLMs to data |
| **Agent/Workflow** | LangGraph, AutoGen, CrewAI | Multi-step, multi-agent, stateful workflows |
| **Prompt Optimization** | DSPy, Instructor | Programmatically optimizing prompts and structured output |
| **Enterprise** | Semantic Kernel | Microsoft ecosystem, Java/.NET support |

---

## 1. LangChain

### What It Is
A Python (and JS) framework for building LLM applications. Provides standard interfaces
for models, prompts, chains, memory, RAG, and agents.

### Core Strengths
- **Largest ecosystem** — 100+ integrations (models, vector stores, tools)
- **LCEL** — clean `|` pipe syntax for composing pipelines
- **RAG-ready** — built-in document loaders, splitters, embeddings, retrievers
- **LangSmith** — observability, tracing, and evaluation platform
- **Most tutorials and community support**

### Weaknesses
- Abstraction can hide what's actually happening (hard to debug)
- Over-engineered for simple tasks
- Breaking changes between versions (0.1 → 0.2 → 0.3)
- `ConversationBufferMemory` and `ConversationChain` deprecated

### Best For
- Linear pipelines: prompt → LLM → output
- RAG applications
- Teams that want a mature ecosystem with lots of examples
- Prototyping quickly

### Code Pattern
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_template("Summarize: {text}") | ChatOpenAI() | StrOutputParser()
result = chain.invoke({"text": "..."})
```

---

## 2. LangGraph

### What It Is
Built ON TOP of LangChain by the same team. Adds stateful, cyclical graph execution
for complex workflows that chains can't express.

### Core Strengths
- **Cycles** — loops that chains fundamentally cannot do
- **Typed state** — all data in a TypedDict, no hidden variables
- **Checkpointing** — save/resume anywhere in the workflow
- **Human-in-the-loop** — built-in interrupt/resume mechanism
- **Multi-agent** — supervisor and worker agents natively
- **Production-ready** — used at scale by enterprise customers

### Weaknesses
- More verbose than LangChain chains
- Steeper learning curve (state machines require planning)
- Overkill for simple single-pass tasks

### Best For
- Agentic workflows that loop until a condition is met
- Multi-agent coordination
- Workflows needing human approval
- Long-running stateful processes
- Customer service agents, research agents, code generation loops

### Code Pattern
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    input: str
    output: str

def process(state: State) -> dict:
    return {"output": "result"}

graph = StateGraph(State)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.add_edge("process", END)
app = graph.compile()
```

---

## 3. LlamaIndex

### What It Is
Originally "GPT Index" — a framework focused exclusively on **connecting LLMs to data**.
While LangChain is a general-purpose framework, LlamaIndex specializes in RAG and
data ingestion.

### Website
`https://www.llamaindex.ai`

### Core Strengths
- **Best-in-class RAG** — more RAG patterns out of the box than LangChain
- **100+ data connectors** — PDF, Notion, Google Drive, Slack, databases
- **Query engines** — structured, unstructured, SQL, multi-hop queries
- **Better default chunking strategies** — HierarchicalNodeParser, SentenceWindowParser
- **Sub-question decomposition** — breaks complex queries into sub-questions
- **LlamaHub** — community connectors and tools marketplace
- **Simpler API** than LangChain for pure RAG use cases

### Weaknesses
- Less general-purpose than LangChain (weaker for non-RAG tasks)
- Smaller agent ecosystem
- Less community support than LangChain

### Best For
- Production RAG systems
- Enterprise document search and Q&A
- Multi-document reasoning
- When you need advanced retrieval (hybrid search, re-ranking, parent-child)

### Code Pattern (RAG)
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents and build index in 3 lines
documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is the refund policy?")
print(response)
```

### LangChain vs LlamaIndex for RAG

| Feature | LangChain | LlamaIndex |
|---|---|---|
| Setup complexity | More boilerplate | Less boilerplate |
| Data connectors | Good (many) | Better (more specialized) |
| Chunking strategies | Basic | Advanced (hierarchical, sentence window) |
| Query patterns | Standard | More options (sub-question, multi-hop) |
| Agent ecosystem | Better | Weaker |
| General use | Excellent | Limited to data tasks |

---

## 4. Haystack

### What It Is
An open-source NLP framework by deepset, focused on production-ready
document search and question answering pipelines.

### Website
`https://haystack.deepset.ai`

### Core Strengths
- **Production-first** design — built for reliability at scale
- **Component-based** architecture — very modular and testable
- **Self-hosted models** — strong support for running models locally (Hugging Face)
- **Evaluation tools** — built-in pipeline evaluation and benchmarking
- **YAML pipelines** — define pipelines as configuration files
- **Document stores** — Elasticsearch, Weaviate, Qdrant support

### Weaknesses
- Steeper learning curve than LangChain
- Smaller community
- Less support for cutting-edge LLM features (agents, function calling)
- More backend/infrastructure-oriented

### Best For
- Enterprise search systems
- Teams that need self-hosted, on-premise solutions
- Compliance-sensitive industries (healthcare, finance) where data can't leave your servers
- Large-scale document retrieval

### Code Pattern
```python
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
pipeline.add_component("generator", OpenAIGenerator())
pipeline.connect("retriever", "generator")

result = pipeline.run({"retriever": {"query": "What is our return policy?"}})
```

---

## 5. Semantic Kernel (Microsoft)

### What It Is
Microsoft's open-source SDK for integrating LLMs into applications.
Available in **Python, C#, and Java** — the only major AI framework with Java support.

### Website
`https://learn.microsoft.com/en-us/semantic-kernel/`

### Core Strengths
- **Java support** — unique in the AI framework space (Java analogy: it's like Spring AI)
- **C# / .NET** — best choice for .NET applications
- **Azure integration** — first-class Azure OpenAI, Azure Cognitive Search
- **Planner** — automatically sequences skills/functions to achieve a goal
- **Enterprise-grade** — Microsoft backing, enterprise support SLAs
- **Plugin/Skill model** — encapsulate AI capabilities as reusable components

### Weaknesses
- Python version less mature than LangChain
- Smaller community than LangChain
- More complex for simple tasks
- Documentation is Microsoft-heavy (Azure-centric examples)

### Best For
- **Java / .NET enterprise applications** ← main differentiation
- Azure cloud environments
- Enterprises with existing Microsoft infrastructure
- Teams needing enterprise support contracts

### Code Pattern (Java)
```java
// Java example — unique to Semantic Kernel
import com.microsoft.semantickernel.*;

Kernel kernel = Kernel.builder()
    .withAIService(ChatCompletionService.class,
        OpenAIChatCompletion.builder()
            .withModelId("gpt-4o")
            .withOpenAIAsyncClient(client)
            .build())
    .build();

var result = kernel.invokePromptAsync("Summarize: {{$input}}")
    .withArguments(KernelFunctionArguments.builder()
        .withVariable("input", "Your text here")
        .build())
    .block();
```

---

## 6. AutoGen (Microsoft)

### What It Is
A framework for building **multi-agent conversational systems** where agents
talk to each other to solve problems collaboratively.

### Website
`https://microsoft.github.io/autogen/`

### Core Strengths
- **Agent conversations** — agents debate, correct each other, collaborate
- **Code execution** — agents can write and run code in a sandbox
- **Group chat** — multiple agents + human in a conversation
- **Easy to create** — define an agent in a few lines
- **Microsoft backing** — active development, research-backed

### Weaknesses
- Less control over agent behavior than LangGraph
- No built-in persistence/checkpointing (less mature)
- Can be unpredictable (agents going off-track)
- Harder to deploy in production

### Best For
- Research and experimentation
- Code generation with auto-testing
- Problem-solving that benefits from multiple perspectives
- Academic / research use cases

### Code Pattern
```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4o"}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "code"}
)

# Agents converse with each other to solve the task
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to sort a list of dicts by a key."
)
```

---

## 7. CrewAI

### What It Is
A framework for orchestrating **role-playing AI agents** that collaborate
like a team (crew) with defined roles, goals, and tasks.

### Website
`https://www.crewai.com`

### Core Strengths
- **Role-based agents** — assign roles like "Researcher", "Writer", "Analyst"
- **Simple API** — easiest multi-agent setup in the ecosystem
- **Task assignment** — assign tasks to agents explicitly
- **Process types** — sequential or hierarchical task execution
- **Rapid prototyping** — working multi-agent system in under 50 lines

### Weaknesses
- Less control than LangGraph (black-box orchestration)
- No built-in loop/cycle support (less flexible than LangGraph)
- Smaller ecosystem
- Less suitable for complex stateful workflows

### Best For
- Quick multi-agent prototypes
- Content creation pipelines (research → write → edit)
- When you want a simple multi-agent setup without learning LangGraph
- Demos and MVPs

### Code Pattern
```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate information about {topic}",
    backstory="Expert researcher with attention to detail",
    llm="gpt-4o"
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging content based on research",
    backstory="Professional writer who creates clear, compelling content",
    llm="gpt-4o"
)

research_task = Task(description="Research {topic}", agent=researcher)
write_task    = Task(description="Write a blog post based on the research", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff(inputs={"topic": "LangGraph"})
```

### LangGraph vs CrewAI

| Feature | LangGraph | CrewAI |
|---|---|---|
| Control | Full control over flow | Abstracted, less control |
| Learning curve | Steeper | Gentle |
| Flexibility | Very high | Medium |
| Loops/cycles | Native support | Limited |
| State management | Explicit TypedDict | Implicit |
| Production readiness | High | Medium |
| Best for | Complex workflows | Quick prototypes |

---

## 8. DSPy (Stanford)

### What It Is
A framework that treats **prompt engineering as a programming problem**.
Instead of manually writing prompts, you define the input/output signature
and DSPy automatically optimizes the prompts.

### Website
`https://dspy-docs.vercel.app`

### Core Strengths
- **Automatic prompt optimization** — no manual prompt engineering needed
- **Composable modules** — build complex pipelines from simple building blocks
- **Optimizer** — BootstrapFewShot, MIPRO automatically find best prompts
- **Declarative** — say WHAT you want, not HOW to prompt for it
- **Research-backed** — Stanford NLP group, strong academic foundation

### Weaknesses
- Very different paradigm — steep learning curve
- Requires labeled training examples for optimization
- Slower development cycle (optimization takes time)
- Less community support

### Best For
- When prompt quality is critical and needs optimization
- Research projects
- Complex reasoning pipelines (chain-of-thought, multi-hop)
- When you want reproducible, programmatically-controlled prompts

### Code Pattern
```python
import dspy

# Define the task signature (not the prompt — DSPy figures out the prompt)
class SentimentClassifier(dspy.Signature):
    """Classify sentiment of customer review."""
    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")

# DSPy automatically creates and optimizes the prompt
classifier = dspy.Predict(SentimentClassifier)
result = classifier(review="The product was amazing, 5 stars!")
print(result.sentiment)  # positive
```

---

## Decision Guide: Which Framework to Use?

```
What are you building?
│
├── Simple prompt → response pipeline?
│   └── LangChain LCEL (prompt | llm | parser)
│
├── Need to search/query your own documents (RAG)?
│   ├── Need maximum simplicity?     → LlamaIndex
│   ├── Need maximum flexibility?    → LangChain
│   └── Need on-premise/self-hosted? → Haystack
│
├── Need complex workflow with loops/cycles?
│   └── LangGraph
│
├── Need multiple AI agents collaborating?
│   ├── Need full control + loops?   → LangGraph (multi-agent)
│   ├── Need quick prototype?        → CrewAI
│   └── Need code execution sandbox? → AutoGen
│
├── Building in Java or .NET?
│   └── Semantic Kernel (ONLY framework with Java support)
│
├── Need production-grade search at scale?
│   └── Haystack
│
└── Need optimized prompts automatically?
    └── DSPy
```

---

## Summary Comparison Table

| Framework | Language | Best For | Difficulty | Community |
|---|---|---|---|---|
| **LangChain** | Python, JS | General LLM apps, RAG | Medium | ⭐⭐⭐⭐⭐ |
| **LangGraph** | Python | Stateful agents, loops | Hard | ⭐⭐⭐⭐ |
| **LlamaIndex** | Python, JS | RAG, document Q&A | Medium | ⭐⭐⭐⭐ |
| **Haystack** | Python | Enterprise search | Hard | ⭐⭐⭐ |
| **Semantic Kernel** | Python, C#, **Java** | Microsoft/Azure, Java | Medium | ⭐⭐⭐ |
| **AutoGen** | Python | Multi-agent, code gen | Medium | ⭐⭐⭐⭐ |
| **CrewAI** | Python | Quick multi-agent | Easy | ⭐⭐⭐⭐ |
| **DSPy** | Python | Prompt optimization | Hard | ⭐⭐⭐ |

---

## For This Course

You are learning **LangChain + LangGraph** — this is the right choice because:
1. Largest community and most job demand
2. LangGraph extends LangChain (not a separate install)
3. Together they cover everything from simple chains to complex agents
4. Production-proven at companies like Replit, Elastic, Rakuten

The others are worth knowing about but LangChain + LangGraph covers 90% of
real-world AI application use cases.
