# Production RAG Chatbot API

A production-grade Retrieval Augmented Generation (RAG) system
built from scratch using LangChain LCEL, FAISS, and FastAPI.

Built by **Sabyasachi Ghosh** — Senior SDET transitioning to Gen AI Engineering.

---

## What It Does

- Upload any PDF document
- Ask questions in natural language
- Get accurate answers grounded in the document
- Returns source page numbers with every answer
- Zero hallucination — answers ONLY from provided context

---

## Live Demo

```
POST /ask
{"question": "What is the EU roaming charge?"}

→ {
    "answer": "The roaming charge for EU/EEA countries is €0.19 per minute for calls.",
    "page_numbers": [1, 2]
  }
```

---

## Architecture

```
PDF
 ↓
Chunk (RecursiveCharacterTextSplitter — 500 tokens, 10% overlap)
 ↓
Embed (OpenAI text-embedding-3-small — 1536 dimensions)
 ↓
FAISS Vector Store (similarity search, top-4 chunks)
 ↓
LangChain LCEL Pipeline
 ↓
GPT-4o-mini (temperature=0, context-only prompt)
 ↓
FastAPI REST Endpoint
 ↓
Answer + Source Page Numbers
```

---

## Why This Is Production Grade

| Feature | Basic Tutorial | This Project |
|---|---|---|
| Pipeline | Legacy RetrievalQA chains | Modern LangChain LCEL |
| Chunking | Fixed size, no overlap | Recursive with 10% overlap |
| Retrieval | Vector search only | Similarity search + source tracking |
| Hallucination | No guardrails | Context-only system prompt |
| API | Terminal script | FastAPI + Pydantic validation |
| Input validation | None | Pydantic BaseModel |
| Output validation | None | response_model strips internal fields |
| Secrets | Hardcoded | .env + .gitignore |

---

## Tech Stack

| Component | Technology |
|---|---|
| Pipeline | LangChain LCEL v0.3+ |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini (temperature=0) |
| API Framework | FastAPI |
| Validation | Pydantic v2 |
| Server | Uvicorn |
| Language | Python 3.10+ |

---

## Project Structure

```
RAG_Core_App_With_Testing/
├── rag.py                  # Core RAG pipeline
│   ├── load_pdf()          # Load and parse PDF pages
│   ├── chunk_documents()   # Split into 500-token chunks with overlap
│   ├── create_vector_store() # Embed chunks and store in FAISS
│   ├── build_rag_chain()   # Build LCEL pipeline with custom prompt
│   └── ask_question()      # Query chain, return answer + source pages
├── app.py                  # FastAPI REST API
│   ├── QuestionRequest     # Pydantic input model
│   ├── AnswerResponse      # Pydantic output model
│   └── POST /ask           # Main endpoint
├── data/
│   └── odido_billing_faq.pdf  # Sample telecom billing document
├── .env.example            # Environment variables template
├── .gitignore              # Protects secrets from git
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/sabyag42/RAG_Core_App_With_Testing.git
cd RAG_Core_App_With_Testing
```

**2. Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

**5. Add your PDF to the data/ folder**

**6. Start the API server**
```bash
uvicorn app:app --reload
```

**7. Open interactive API docs**
```
http://localhost:8000/docs
```

---

## API Reference

### POST /ask

Ask a question about the loaded document.

**Request Body:**
```json
{
  "question": "string"
}
```

**Response:**
```json
{
  "answer": "string",
  "page_numbers": [1, 2]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"How do I cancel my plan?\"}"
```

**Response:**
```json
{
  "answer": "You can cancel your plan via the My Odido App, by calling 1234, online at odido.nl/cancel, or in store with a valid photo ID. A 30-day notice period applies.",
  "page_numbers": [2]
}
```

---

## How the LCEL Pipeline Works

```python
rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,  # retrieve chunks → format as string
        "question": RunnablePassthrough()     # pass question unchanged
    })
    | prompt          # fill system prompt template with context + question
    | llm             # GPT-4o-mini generates grounded answer
    | StrOutputParser()  # extract clean text string from AIMessage
)
```

Each step is a Runnable — composable, streamable, and swappable.
Replace `llm` with Claude or LLaMA without changing anything else.

---

## Key Design Decisions

**Why RecursiveCharacterTextSplitter?**
Tries to split by paragraphs first, then sentences, then words.
Preserves natural document structure. Never cuts mid-sentence if avoidable.

**Why 500 tokens with 10% overlap?**
500 tokens = enough context per chunk without including irrelevant content.
50-token overlap ensures boundary sentences appear in both adjacent chunks.

**Why temperature=0?**
This is a factual Q&A system. We want deterministic, consistent answers.
Temperature=0 means the model always picks the highest-probability token.

**Why return_source_documents / source pages?**
Transparency reduces hallucination trust issues.
Users can verify answers against the original document.
Also satisfies GDPR explainability requirements for AI decisions.

---

## Roadmap

- [x] Core RAG pipeline with LCEL
- [x] FastAPI REST endpoint with Pydantic validation
- [ ] RAGAS automated evaluation pipeline
- [ ] A/B testing framework for prompt comparison
- [ ] GitHub Actions CI/CD — eval gates on every push
- [ ] Hybrid retrieval (FAISS + BM25 keyword search)
- [ ] Streamlit frontend UI
- [ ] Docker containerisation

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key from platform.openai.com |

---

## Connect

- GitHub: [github.com/sabyag42](https://github.com/sabyag42)
- LinkedIn: [Sabyasachi Ghosh](https://www.linkedin.com/in/sabyasachi-ghosh-sg/)