# LangChain Mastery — 30-Day Gen AI Engineering Sprint

A hands-on portfolio built module by module, covering LangChain,
LangGraph, RAG, MCP, and production deployment.

Every module is a working project with clean code, LangSmith
tracing, and a real-world use case.

Built by **Sabyasachi Ghosh** — Senior SDET transitioning to
Gen AI Engineering.

---

## Modules

| # | Project | Concepts | Status |
|---|---------|----------|--------|
| 1 | [AI Q&A CLI Tool](./module1-qa-cli) | LCEL · prompts · chains · StrOutputParser | ✅ Done |
| 2 | [Bug Report Extractor](./module2-bug-extractor) | Pydantic · structured output · PydanticOutputParser | ✅ Done |
| 3 | [LangGraph Joke Bot](./module3-langgraph-basics) | StateGraph · nodes · edges · shared state | ✅ Done |
| 4 | [Picky Joke Bot](./module4-conditional-edges) | Conditional edges · routing · retry loops | 🔄 In progress |
| 5 | Coming soon | ReAct agents · tool calling | ⏳ |
| 6 | Coming soon | Memory · checkpointing · persistence | ⏳ |
| 7 | Coming soon | Multi-agent · supervisor pattern | ⏳ |
| 8 | Coming soon | RAG · embeddings · vector stores | ⏳ |
| 9 | Coming soon | MCP · servers · clients | ⏳ |
| 10 | Coming soon | Production · FastAPI · Docker · CI/CD | ⏳ |

---

## Tech Stack

- **Orchestration:** LangGraph 1.0 · LangChain 0.3.x
- **LLM:** OpenAI GPT-4o-mini
- **Observability:** LangSmith (tracing + evaluation)
- **Package management:** uv
- **Deployment:** FastAPI · Docker · GitHub Actions · Railway
- **Language:** Python 3.11+

---

## Why this portfolio

Most Gen AI portfolios show one ChatGPT wrapper. This one shows:

- **Depth** — from basic LCEL chains to multi-agent LangGraph systems
- **Production thinking** — every project has tracing, error handling,
  and deployment
- **SDET angle** — testable nodes, evaluation metrics, observability
- **Current API** — LangGraph v1.0 canonical patterns, April 2026

---

## Running any module

Each module is a self-contained uv project:

```bash
cd moduleX-name
uv sync
# Add your API keys to .env
uv run python src/graph.py
```

---

## Connect

- GitHub: [github.com/sabyag42](https://github.com/sabyag42)
- LinkedIn: https://www.linkedin.com/in/sabyasachi-ghosh-sg/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B3Tm2qg%2BxQyiqAg4v743DqQ%3D%3D
