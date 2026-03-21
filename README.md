# AI Engineering — Complete Study Guide
### From Zero to Production-Ready AI Engineer

---

> **Built for career transitioners.**
> Whether you are a developer, analyst, or complete beginner — this repo is a structured, week-by-week roadmap to becoming an AI Engineer.
> Follow the weekly schedule in order. Every folder = one week of focused learning.

---

## Why This Repo?

The AI engineering field moves fast. Most learning resources are either too shallow (blog posts) or too academic (research papers). This guide sits in the middle — **practical, structured, and production-focused**.

By the end of this roadmap you will be able to:

- Build and deploy LLM-powered applications
- Design RAG systems over your own documents
- Create multi-agent AI workflows using LangGraph and AutoGen
- Deploy AI systems on Docker, Kubernetes, Azure, and AWS
- Write production-ready AI code with monitoring, CI/CD, and observability

> **This is a living repo.** Notes are added week by week. Star and watch to follow along.

---

## Progress Tracker

```
Generative AI Path   [█░░░░░░░░░░░░░]  Week 1 of 14  — In Progress
Agentic AI Path      [░░░░░░░░░░░░░░]  Week 15 of 36  — Not Started
```

| Path | Weeks | Done | Remaining |
|---|---|---|---|
| Generative AI | 14 weeks | 1 | 13 |
| Agentic AI | 22 weeks | 0 | 22 |

---

## How to Use This Repo

```
Step 1 → Start with week-1 and work forward in order
Step 2 → Use 00_AI_Engineering_Master_Guide.md as your reference at any time
Step 3 → Practice each lab before moving to the next week
Step 4 → Push your own notes and labs to your fork

Status legend:
  ✅  Notes available — read, study, and practice
  ⏳  Coming soon — will be added as the course progresses
```

### Who Should Follow This?

| Background | What You Will Gain |
|---|---|
| Software Developer | Learn to build AI-powered products |
| Data Analyst | Move from analysis to building AI systems |
| Backend Engineer | Add LLMs and agents to your stack |
| Complete Beginner | Structured path from zero to AI Engineer |
| Career Transitioner | Roadmap with projects to show employers |

---

## Master Reference Guide

> One file covering everything. Use it as a quick-lookup companion throughout the journey.

| File | Contents |
|---|---|
| [00_AI_Engineering_Master_Guide.md](00_AI_Engineering_Master_Guide.md) | 36 sections — AI history · ML types · NLP · Python for AI · Neural networks · Transformers · LLMs · Prompt engineering · RAG · Fine-tuning · LangGraph · AutoGen · Docker · Kubernetes · CI/CD · Cloud deployment · Ethics · 100+ term glossary |

---

## Path 1 — Generative AI (Weeks 1–14)

> Covers the full stack from ML fundamentals to deploying LLM applications.

---

### Week 1 — AI & ML History | `week-1-ai-ml-history/`

**What you will learn:** Where AI came from, what ML really means, the 5 types of machine learning, and how to set up Python environments properly.

| File | Topic | Status |
|---|---|---|
| [01_AI_ML_Introduction.md](week-1-ai-ml-history/01_AI_ML_Introduction.md) | History of AI, what is AI/ML/DL/GenAI, how ML evolved | ✅ Done |
| [02_Types_of_Machine_Learning.md](week-1-ai-ml-history/02_Types_of_Machine_Learning.md) | Supervised, Unsupervised, Semi-supervised, RL, Self-supervised | ✅ Done |
| [03_Python_Virtual_Environments.md](week-1-ai-ml-history/03_Python_Virtual_Environments.md) | venv, conda, project structure, .gitignore | ✅ Done |

**Key concepts:** Alan Turing → Deep Blue → AlexNet → Transformers → ChatGPT · Training vs inference · Labels, features, models

---

### Week 2 — AI vs GenAI & NLP Basics | `week-2-ai-genai-nlp/`

**What you will learn:** The difference between traditional AI and Generative AI, and the foundations of Natural Language Processing.

| File | Topic | Status |
|---|---|---|
| `01_AI_vs_GenAI.md` | Key differences, use cases, when to use which | ⏳ Upcoming |
| `02_NLP_Fundamentals.md` | Tokenization, stemming, TF-IDF, word embeddings, NLP pipeline | ⏳ Upcoming |
| `03_Lab_Sentiment_Analysis.md` | Build a sentiment classifier with classical NLP | ⏳ Upcoming |

**Key concepts:** Generative vs discriminative models · Bag of words · TF-IDF · Word2Vec · NLP tasks

---

### Week 3 — Python for AI | `week-3-python-for-ai/`

**What you will learn:** The Python toolkit every AI engineer uses daily — NumPy, Pandas, Regex, REST APIs, and web scraping.

| File | Topic | Status |
|---|---|---|
| `01_NumPy_Pandas.md` | Arrays, DataFrames, data manipulation at scale | ⏳ Upcoming |
| `02_Regex_Text_Processing.md` | Pattern matching for cleaning and extracting text | ⏳ Upcoming |
| `03_REST_APIs_JSON.md` | Calling LLM APIs, handling JSON responses | ⏳ Upcoming |
| `04_Web_Scraping.md` | BeautifulSoup, Requests, scraping for NLP datasets | ⏳ Upcoming |
| `05_Lab_Web_Data_NLP.md` | Collect and analyse web data for an NLP task | ⏳ Upcoming |

**Key concepts:** Vectorised operations · DataFrame joins · HTTP methods · JSON parsing · Robots.txt

---

### Week 4 — Deep Learning & Neural Networks | `week-4-deep-learning/`

**What you will learn:** How neural networks actually work — from a single neuron to multi-layer networks, activation functions, and the training loop.

| File | Topic | Status |
|---|---|---|
| `01_Neural_Network_Basics.md` | Neurons, layers, weights, bias, activation functions | ⏳ Upcoming |
| `02_RNN_LSTM.md` | Recurrent networks for sequences and time series | ⏳ Upcoming |
| `03_Training_Backprop.md` | Loss functions, backpropagation, gradient descent | ⏳ Upcoming |

**Key concepts:** ReLU · Sigmoid · Softmax · Cross-entropy loss · Overfitting vs underfitting · Dropout

---

### Week 5 — Transformers & Attention | `week-5-transformers/`

**What you will learn:** The architecture behind every modern LLM — self-attention, positional encoding, BERT, GPT, and T5.

| File | Topic | Status |
|---|---|---|
| `01_Attention_Mechanism.md` | Self-attention, multi-head attention, Q/K/V | ⏳ Upcoming |
| `02_Transformer_Architecture.md` | Encoder, decoder, positional encoding, residuals | ⏳ Upcoming |
| `03_BERT_GPT_T5.md` | Encoder-only vs decoder-only vs encoder-decoder | ⏳ Upcoming |
| `04_Lab_Text_Classifier.md` | Build a transformer-based text classifier | ⏳ Upcoming |

**Key concepts:** Attention Is All You Need · Masked LM · Causal LM · Context window · Embeddings

---

### Week 6 — Generative AI Concepts | `week-6-generative-ai-concepts/`

**What you will learn:** What GenAI really is, how LLMs are structured, and how diffusion models and GANs generate images.

| File | Topic | Status |
|---|---|---|
| `01_What_is_GenAI.md` | GenAI overview, LLM architecture, why now | ⏳ Upcoming |
| `02_Diffusion_Models.md` | How diffusion works, DALL-E, Stable Diffusion | ⏳ Upcoming |
| `03_GANs_Overview.md` | Generator vs Discriminator, applications | ⏳ Upcoming |
| `04_Lab_Text_Generation.md` | Generate text with a pre-trained GPT model | ⏳ Upcoming |

**Key concepts:** Tokenization · Pre-training · Denoising · Adversarial training · Foundation models

---

### Week 7 — Prompt Engineering | `week-7-prompt-engineering/`

**What you will learn:** The art and science of communicating with LLMs — zero-shot, few-shot, chain-of-thought, and ReAct patterns.

| File | Topic | Status |
|---|---|---|
| `01_Prompt_Types.md` | Zero-shot, few-shot, CoT, ReAct | ⏳ Upcoming |
| `02_Prompt_Templates.md` | Reusable templates for different use cases | ⏳ Upcoming |
| `03_System_Role_Prompting.md` | System instructions, personas, output formats | ⏳ Upcoming |
| `04_Lab_Effective_Prompts.md` | Hands-on prompt writing and iteration | ⏳ Upcoming |

**Key concepts:** Temperature · Top-p · System prompt · Prompt injection · Structured outputs

---

### Week 8 — OpenAI & LLM APIs | `week-8-llm-apis/`

**What you will learn:** How to call LLM APIs in production — GPT-4, DALL-E, Whisper, function calling, and token cost management.

| File | Topic | Status |
|---|---|---|
| `01_OpenAI_API.md` | Chat completions, streaming, parameters | ⏳ Upcoming |
| `02_DALLE_Whisper.md` | Text-to-image generation and speech-to-text | ⏳ Upcoming |
| `03_Function_Calling.md` | Structured JSON output, tool use | ⏳ Upcoming |
| `04_Token_Management.md` | Counting tokens, estimating and reducing cost | ⏳ Upcoming |
| `05_Lab_QA_Bot.md` | Build a Q&A bot with the ChatGPT API | ⏳ Upcoming |

**Key concepts:** API keys · Rate limits · Retry logic · Cost optimisation · Multimodal inputs

---

### Week 9 — LangChain | `week-9-langchain/`

**What you will learn:** The most widely used LLM framework — chains, tools, memory, agents, and RAG with LangChain.

| File | Topic | Status |
|---|---|---|
| `01_LangChain_Intro.md` | Setup, core abstractions, LLM wrappers | ⏳ Upcoming |
| `02_Chains_Tools.md` | Building chains, adding tools | ⏳ Upcoming |
| `03_Memory.md` | Buffer, summary, entity, and vector memory | ⏳ Upcoming |
| `04_Agents_RAG.md` | Agents, retrievers, vector stores | ⏳ Upcoming |
| `05_Lab_Document_Chatbot.md` | Document-aware chatbot with FAISS | ⏳ Upcoming |

**Key concepts:** LCEL · PromptTemplate · OutputParser · AgentExecutor · Retriever

---

### Week 10 — Hugging Face Transformers | `week-10-hugging-face/`

**What you will learn:** Load, run, and fine-tune pre-trained models from the world's largest model hub.

| File | Topic | Status |
|---|---|---|
| `01_HuggingFace_Intro.md` | Hub overview, pipelines, loading models | ⏳ Upcoming |
| `02_Tokenization_Inference.md` | Tokenizers, model inference, batch processing | ⏳ Upcoming |
| `03_Fine_Tuning_Basics.md` | Fine-tune a model on your own data | ⏳ Upcoming |
| `04_Lab_Sentiment_Classifier.md` | End-to-end sentiment classifier | ⏳ Upcoming |

**Key concepts:** AutoTokenizer · AutoModel · Pipeline API · PEFT · LoRA · Model cards

---

### Week 11 — Vector Databases & RAG | `week-11-vector-db-rag/`

**What you will learn:** Semantic search infrastructure — how embeddings work, which vector DB to choose, and how to build a full RAG pipeline.

| File | Topic | Status |
|---|---|---|
| `01_Vector_DBs.md` | FAISS, Chroma, Pinecone — how they work and when to use each | ⏳ Upcoming |
| `02_Embeddings.md` | Embedding models, cosine similarity, semantic search | ⏳ Upcoming |
| `03_RAG_Pipeline.md` | Full RAG system — chunking, indexing, retrieval, generation | ⏳ Upcoming |
| `04_Lab_PDF_Search.md` | Build vector search over PDF documents | ⏳ Upcoming |

**Key concepts:** Chunking strategies · k-NN search · Context recall · Answer faithfulness · RAGAs

---

### Week 12 — Real-World GenAI Projects | `week-12-genai-projects/`

**What you will learn:** Build four complete, portfolio-ready GenAI projects from scratch.

| File | Project | Status |
|---|---|---|
| `01_Resume_Screener.md` | LLM-powered resume screening tool | ⏳ Upcoming |
| `02_FAQ_Chatbot.md` | AI chatbot with FAQ knowledge base | ⏳ Upcoming |
| `03_Chat_with_PDF.md` | Full RAG chatbot over PDF documents | ⏳ Upcoming |
| `04_Text_to_Image.md` | Text-to-image app using DALL-E API | ⏳ Upcoming |

**Portfolio tip:** Each project here belongs in your GitHub and LinkedIn. Show the full build — not just the code.

---

### Week 13 — MLOps & Deployment | `week-13-mlops-deployment/`

**What you will learn:** Take an AI app from a Jupyter notebook to a deployed web application.

| File | Topic | Status |
|---|---|---|
| `01_Streamlit_Gradio.md` | Build AI demo UIs in minutes | ⏳ Upcoming |
| `02_Deploy_HuggingFace_Spaces.md` | Free, instant deployment for ML demos | ⏳ Upcoming |
| `03_Docker_Basics.md` | Package and containerise an AI app | ⏳ Upcoming |
| `04_Lab_Deploy_GenAI_App.md` | Deploy a complete GenAI app to the web | ⏳ Upcoming |

**Key concepts:** Container images · Port mapping · Environment variables · CI/CD basics

---

### Week 14 — Responsible AI & Ethics | `week-14-responsible-ai/`

**What you will learn:** The safety and ethics fundamentals every AI engineer must know before shipping to production.

| File | Topic | Status |
|---|---|---|
| `01_Hallucination_Bias.md` | Types, causes, and mitigation strategies | ⏳ Upcoming |
| `02_Guardrails_Moderation.md` | Content filters, moderation APIs, guardrails | ⏳ Upcoming |
| `03_Copyright_Fairness.md` | IP, watermarking, and fair use in AI | ⏳ Upcoming |
| `04_Lab_Hallucination_Analysis.md` | Measure and reduce hallucination in practice | ⏳ Upcoming |

**Key concepts:** Bias types · Prompt injection · Constitutional AI · Alignment · Privacy

---

## Path 2 — Agentic AI (Weeks 15–36)

> Goes deeper — autonomous agents, multi-agent systems, cloud deployment, and production-grade observability.
> **Prerequisite:** Complete Path 1 first, or have equivalent LLM engineering experience.

---

### Week 15 — Intro to Agentic AI | `week-15-agentic-ai-intro/`

**What you will learn:** What makes an AI system "agentic", the key frameworks, and how to choose between them.

| File | Topic | Status |
|---|---|---|
| `01_What_is_Agentic_AI.md` | Agents, tasks, graphs, the agentic loop | ⏳ Upcoming |
| `02_Framework_Comparison.md` | LangGraph vs AutoGen vs CrewAI vs Semantic Kernel | ⏳ Upcoming |
| `03_LLM_Provider_Comparison.md` | OpenAI vs Azure OpenAI vs AWS Bedrock | ⏳ Upcoming |

---

### Week 16 — LangGraph Basics | `week-16-langgraph-basics/`

**What you will learn:** LangGraph architecture and how to build your first stateful agent workflow.

| File | Topic | Status |
|---|---|---|
| `01_LangGraph_Architecture.md` | Why LangGraph, graphs vs chains | ⏳ Upcoming |
| `02_Environment_Setup.md` | Install, configure, first run | ⏳ Upcoming |
| `03_Lab_First_DAG.md` | Build your first LangGraph DAG | ⏳ Upcoming |

---

### Week 17 — LangGraph Core | `week-17-langgraph-core/`

**What you will learn:** The fundamental building blocks — nodes, edges, state, and conditional routing.

| File | Topic | Status |
|---|---|---|
| `01_Nodes_Edges_State.md` | StateGraph, TypedDict, nodes, edges | ⏳ Upcoming |
| `02_Conditional_Routing.md` | Branch on state, loop back, retry | ⏳ Upcoming |
| `03_Lab_Agentic_Task_Flow.md` | Build a simple agentic task flow | ⏳ Upcoming |

---

### Week 18 — LangGraph Python SDK | `week-18-langgraph-sdk/`

**What you will learn:** Deep dive into the SDK — define nodes, reactive transitions, and test agent components.

| File | Topic | Status |
|---|---|---|
| `01_Python_SDK_Deep_Dive.md` | StateGraph, compile, invoke, stream | ⏳ Upcoming |
| `02_Reactive_Transitions.md` | Handlers, conditional edges, checkpointing | ⏳ Upcoming |
| `03_Testing_Agents.md` | Unit testing strategy for agent nodes | ⏳ Upcoming |

---

### Week 19 — Multi-Agent with LangGraph | `week-19-multi-agent-langgraph/`

**What you will learn:** Build systems where multiple specialised agents interact through shared graph state.

| File | Topic | Status |
|---|---|---|
| `01_Multi_Agent_State.md` | Multiple agents interacting via graph state | ⏳ Upcoming |
| `02_Dynamic_Task_Allocation.md` | Supervisor assigns tasks to workers | ⏳ Upcoming |
| `03_Conditional_Loops.md` | Retry, reflection, and iterative refinement | ⏳ Upcoming |

---

### Week 20 — AutoGen | `week-20-autogen/`

**What you will learn:** Microsoft's AutoGen framework for building conversational multi-agent systems.

| File | Topic | Status |
|---|---|---|
| `01_AutoGen_Intro.md` | Architecture, agent types, AutoGen vs LangGraph | ⏳ Upcoming |
| `02_Custom_Agents.md` | Roles, communication protocols, tool integration | ⏳ Upcoming |
| `03_Supervisor_Agents.md` | Helper agents and supervisor pattern | ⏳ Upcoming |
| `04_Multi_Agent_Chat.md` | GroupChat, speaker selection methods | ⏳ Upcoming |

---

### Week 21 — AutoGen + LangGraph Combined | `week-21-autogen-langgraph/`

**What you will learn:** Orchestrate AutoGen agents inside LangGraph for the best of both frameworks.

| File | Topic | Status |
|---|---|---|
| `01_Combining_Frameworks.md` | Run AutoGen conversations as LangGraph nodes | ⏳ Upcoming |
| `02_Multi_Turn_Conversations.md` | Handle long stateful conversations | ⏳ Upcoming |
| `03_Error_Handling.md` | Retry logic, fallback agents, edge case design | ⏳ Upcoming |

---

### Week 22 — Real-World Agent Pipelines | `week-22-real-world-pipelines/`

**What you will learn:** Apply agentic AI to real document processing problems — invoice parsing and OCR.

| File | Topic | Status |
|---|---|---|
| `01_Invoice_Parsing.md` | Agents for document interpretation | ⏳ Upcoming |
| `02_OCR_Pipeline.md` | Azure Vision, AWS Textract, Tesseract | ⏳ Upcoming |
| `03_Data_Validation_Agents.md` | Retry agents, fallback design patterns | ⏳ Upcoming |

---

### Week 23 — Docker for AI Agents | `week-23-docker-agents/`

**What you will learn:** Package and run AI agents as portable, reproducible containers.

| File | Topic | Status |
|---|---|---|
| `01_Docker_Basics.md` | Images, containers, volumes, networks | ⏳ Upcoming |
| `02_Dockerfile_AI.md` | Write Dockerfiles for Python AI agents | ⏳ Upcoming |
| `03_Docker_Compose.md` | Multi-service setup: agent + vector DB + cache | ⏳ Upcoming |
| `04_Lab_Dockerized_LangGraph.md` | Containerise and test a LangGraph app | ⏳ Upcoming |

---

### Week 24 — Kubernetes for AI | `week-24-kubernetes/`

**What you will learn:** Deploy and scale AI agents on Kubernetes — the industry standard for production containers.

| File | Topic | Status |
|---|---|---|
| `01_K8s_Basics.md` | Pods, Deployments, Services, ConfigMaps | ⏳ Upcoming |
| `02_K8s_vs_Docker_Compose.md` | When to use each | ⏳ Upcoming |
| `03_LangGraph_in_K8s.md` | Write manifests and deploy a LangGraph pipeline | ⏳ Upcoming |
| `04_Scaling_Monitoring.md` | HPA autoscaling, Prometheus, Grafana | ⏳ Upcoming |

---

### Week 25 — Azure Cloud Deployment | `week-25-azure-deployment/`

**What you will learn:** Deploy production AI applications on Microsoft Azure.

| File | Topic | Status |
|---|---|---|
| `01_Azure_Overview.md` | Resource groups, App Service, Container Registry | ⏳ Upcoming |
| `02_Azure_OpenAI.md` | API setup, authentication, quota handling | ⏳ Upcoming |
| `03_Lab_Deploy_to_Azure.md` | Deploy a Docker container to Azure Web App | ⏳ Upcoming |

---

### Week 26 — AWS Cloud Deployment | `week-26-aws-deployment/`

**What you will learn:** Deploy AI applications on AWS using ECS Fargate, Bedrock, and Textract.

| File | Topic | Status |
|---|---|---|
| `01_AWS_Overview.md` | ECS, Fargate, ECR, IAM fundamentals | ⏳ Upcoming |
| `02_AWS_Bedrock.md` | Access Claude, Llama, and Titan via Bedrock | ⏳ Upcoming |
| `03_Textract_OCR.md` | Document text extraction with AWS Textract | ⏳ Upcoming |
| `04_Lab_Deploy_to_AWS.md` | Deploy a LangGraph app on ECS Fargate | ⏳ Upcoming |

---

### Week 27 — CI/CD for AI | `week-27-cicd-ai/`

**What you will learn:** Automate the build, test, and deploy pipeline for AI applications using GitHub Actions.

| File | Topic | Status |
|---|---|---|
| `01_GitHub_Actions.md` | CI/CD basics for AI projects | ⏳ Upcoming |
| `02_Docker_CI_Pipeline.md` | Build, test, and push image on every push | ⏳ Upcoming |
| `03_K8s_Deploy_Pipeline.md` | Auto-deploy to Kubernetes on merge to main | ⏳ Upcoming |

---

### Week 28 — Production & Observability | `week-28-production-observability/`

**What you will learn:** Make AI systems production-ready — rate limiting, key management, tracing, and cost tracking.

| File | Topic | Status |
|---|---|---|
| `01_Production_Patterns.md` | Rate limiting, retries, API Gateway | ⏳ Upcoming |
| `02_Secure_Key_Management.md` | Secrets in K8s, Azure Key Vault, AWS Secrets Manager | ⏳ Upcoming |
| `03_OpenTelemetry_Tracing.md` | Distributed tracing for long-running agent flows | ⏳ Upcoming |
| `04_Performance_Benchmarking.md` | Token analysis, latency, cost-performance balance | ⏳ Upcoming |

---

### Week 29 — Advanced Agents & Feedback | `week-29-advanced-agents/`

**What you will learn:** Push agent quality higher with advanced prompting, feedback loops, and self-healing behaviour.

| File | Topic | Status |
|---|---|---|
| `01_Advanced_Prompt_Engineering.md` | ReAct, CoT, structured outputs for agents | ⏳ Upcoming |
| `02_Feedback_Loops.md` | Capture and act on agent output feedback | ⏳ Upcoming |
| `03_Self_Healing_Agents.md` | Feedback loops, dynamic policy adjustment | ⏳ Upcoming |
| `04_Agent_Behavior_Tuning.md` | Prompt templates, personality config, memory tradeoffs | ⏳ Upcoming |

---

### Week 30 — Testing & Simulation | `week-30-testing-simulation/`

**What you will learn:** Test AI systems rigorously — end-to-end pipelines, A/B experiments, and synthetic data.

| File | Topic | Status |
|---|---|---|
| `01_E2E_Pipeline_Testing.md` | Testing full agentic pipelines | ⏳ Upcoming |
| `02_AB_Testing.md` | Compare agent versions with experiments | ⏳ Upcoming |
| `03_Synthetic_Data.md` | Generate test data for edge cases | ⏳ Upcoming |

---

### Weeks 31–36 — Capstone Projects | `week-31-capstone/`

**What you will build:** Six real-world AI agent applications — each one portfolio-ready and production-deployable.

| Project | What It Does |
|---|---|
| AI Travel Planner Agent | Takes user preferences → builds full travel itinerary autonomously |
| Customer Support AI Agent | Answers FAQs + escalates + updates tickets + sends emails |
| Personal Finance Assistant | Connects to expense data → categorises → advises on budget |
| E-Commerce Product Finder | Searches multiple sites → compares → recommends best option |
| Research Automation Agent | Scans articles → summarises → generates structured insights |
| Workflow Automation Agent | Integrates Slack + Gmail + Trello → schedules, assigns, reminds |

> These projects demonstrate the full AI engineering stack. Use them as your portfolio for job applications and interviews.

---

## Folder Structure

```
generative-ai-notes/
│
├── README.md                              ← You are here
├── 00_AI_Engineering_Master_Guide.md      ← Complete reference (use anytime)
│
│   ── PATH 1: GENERATIVE AI ──────────────────────────────────
│
├── week-1-ai-ml-history/                  ✅ Done
│   ├── 01_AI_ML_Introduction.md
│   ├── 02_Types_of_Machine_Learning.md
│   └── 03_Python_Virtual_Environments.md
│
├── week-2-ai-genai-nlp/                   ⏳ Coming soon
├── week-3-python-for-ai/                  ⏳ Coming soon
├── week-4-deep-learning/                  ⏳ Coming soon
├── week-5-transformers/                   ⏳ Coming soon
├── week-6-generative-ai-concepts/         ⏳ Coming soon
├── week-7-prompt-engineering/             ⏳ Coming soon
├── week-8-llm-apis/                       ⏳ Coming soon
├── week-9-langchain/                      ⏳ Coming soon
├── week-10-hugging-face/                  ⏳ Coming soon
├── week-11-vector-db-rag/                 ⏳ Coming soon
├── week-12-genai-projects/                ⏳ Coming soon
├── week-13-mlops-deployment/              ⏳ Coming soon
├── week-14-responsible-ai/                ⏳ Coming soon
│
│   ── PATH 2: AGENTIC AI ─────────────────────────────────────
│
├── week-15-agentic-ai-intro/              ⏳ Coming soon
├── week-16-langgraph-basics/              ⏳ Coming soon
├── week-17-langgraph-core/               ⏳ Coming soon
├── week-18-langgraph-sdk/                 ⏳ Coming soon
├── week-19-multi-agent-langgraph/         ⏳ Coming soon
├── week-20-autogen/                       ⏳ Coming soon
├── week-21-autogen-langgraph/             ⏳ Coming soon
├── week-22-real-world-pipelines/          ⏳ Coming soon
├── week-23-docker-agents/                 ⏳ Coming soon
├── week-24-kubernetes/                    ⏳ Coming soon
├── week-25-azure-deployment/              ⏳ Coming soon
├── week-26-aws-deployment/                ⏳ Coming soon
├── week-27-cicd-ai/                       ⏳ Coming soon
├── week-28-production-observability/      ⏳ Coming soon
├── week-29-advanced-agents/               ⏳ Coming soon
├── week-30-testing-simulation/            ⏳ Coming soon
└── week-31-capstone/                      ⏳ Coming soon
```

---

## Tools & Technologies Covered

| Category | Tools |
|---|---|
| **Languages** | Python |
| **Data** | NumPy, Pandas, Regex, BeautifulSoup |
| **ML Frameworks** | PyTorch, TensorFlow, scikit-learn, XGBoost |
| **LLM APIs** | OpenAI GPT-4, DALL-E, Whisper, Anthropic Claude |
| **LLM Orchestration** | LangChain, LangGraph, LlamaIndex |
| **Multi-Agent** | AutoGen, CrewAI, Microsoft Semantic Kernel |
| **Hugging Face** | Transformers, PEFT, Datasets, Spaces |
| **Vector Databases** | FAISS, Chroma, Pinecone, Weaviate, Milvus |
| **UI / Demo** | Streamlit, Gradio, Chainlit |
| **Experiment Tracking** | MLflow, Weights & Biases |
| **Serving** | FastAPI, BentoML, Triton |
| **Monitoring** | Prometheus, Grafana, OpenTelemetry, Evidently AI |
| **Containers** | Docker, Docker Compose |
| **Orchestration** | Kubernetes, Helm, Minikube |
| **Cloud** | Azure App Service, Azure OpenAI, AWS ECS, AWS Bedrock, AWS Textract |
| **CI/CD** | GitHub Actions |
| **OCR** | Azure Cognitive Vision, AWS Textract, Tesseract |

---

## Skills You Will Have After This Roadmap

```
After Week 6:   You understand how LLMs work — not just how to use them
After Week 9:   You can build LangChain-powered AI applications
After Week 12:  You have 4 portfolio projects to show employers
After Week 14:  You can ship GenAI apps to production responsibly
After Week 21:  You can build multi-agent systems with LangGraph + AutoGen
After Week 28:  You can deploy, monitor, and maintain AI at scale
After Week 36:  You are an AI Engineer — ready for the job market
```

---

## Weekly Update Log

| Week | Folder | Topics Added |
|---|---|---|
| Week 1 | `week-1-ai-ml-history/` | History of AI, Types of ML, Python Virtual Environments |

---

## Contributing

Found an error or want to add something? Open a PR or raise an issue.
If this repo helped you, give it a star — it helps others find it too.

---

*Notes added weekly. Follow this repo to learn AI Engineering step by step.*
