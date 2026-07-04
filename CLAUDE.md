# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is a monorepo of independent, unrelated AI experiments and learning projects ("AI Playground"). There is no shared build system, dependency file, or CI across projects — each top-level directory is its own self-contained project with its own stack. Treat each one in isolation; changes in one do not affect the others.

- `MCP/GDS-MCP/` — TypeScript MCP server (Node/vitest)
- `RAG/` — Python LangGraph agentic RAG system
- `Models/` — standalone ML experiments (neural net from scratch, a CLIP/ViT notebook)
- `Skills_and_prompts/` — Claude Code slash commands and skill/prompt docs (no code)

Always check which subdirectory you're working in before assuming a command or convention applies — e.g. don't run `npm test` for the RAG project or `pytest` for GDS-MCP.

## Releases

Releases are automated with semantic-release (`.github/workflows/release.yml` + `.releaserc.json`): every push to `main` scans commits since the last tag, and a releasable commit produces a git tag, an updated `CHANGELOG.md` committed back to `main`, and a GitHub Release. Commit messages (or squash-merge PR titles) must follow Conventional Commits to trigger a release: `fix:` → patch, `feat:` → minor, `feat!:` or a `BREAKING CHANGE` footer → major. Other prefixes (`chore:`, `docs:`, etc.) and non-conventional messages release nothing.

## MCP/GDS-MCP (TypeScript MCP server)

Read-only MCP server that helps coding agents generate/validate GOV.UK Design System markup correctly. Has its own `CLAUDE.md` — read `MCP/GDS-MCP/CLAUDE.md` before working here, it documents the deep-module architecture (knowledge/ vs tools/ vs templates/) and the exact steps for adding a new component.

```bash
cd MCP/GDS-MCP
npm install
npm test          # vitest watch mode — 104 tests
npm run test:run  # vitest single run
npm run dev        # start server on stdio (tsx)
npm run build      # compile to dist/
npm start          # run compiled server
```

Run a single test file: `npx vitest run tests/tools/suggest-component.test.ts`

Key architecture point: three thin MCP tools (`suggest_component`, `generate_markup`, `review_html`) sit on top of a deep `src/knowledge/` module (34 GOV.UK component definitions + accessibility/design-pattern rules). No HTML parser is used for `review_html` — rules use regex/string matching since GOV.UK markup is predictable. No runtime dependencies beyond the MCP SDK and Zod; all component knowledge is baked in (no external API calls).

## RAG (LangGraph agentic RAG)

A LangGraph state-machine implementation of "Adaptive/Corrective RAG": routes a question to either a Pinecone vectorstore or Tavily web search, grades retrieved documents for relevance, generates an answer, then grades that answer for hallucination and for actually answering the question — looping back to web search or regeneration if it fails those checks.

There is no `requirements.txt`; install manually per the README:
```bash
cd RAG
pip install langchain langchainhub langchain-community langchain-tavily langchain-pinecone langgraph python-dotenv pytest langchain-openai
```

Requires a `.env` with `OPENAI_API_KEY`, `PINECONE_API_KEY`, `TAVILY_API_KEY` (optional `LANGSMITH_*` for tracing). Tests and ingestion make live calls to OpenAI/Pinecone/Tavily — there is no mocking layer.

```bash
python ingestion.py   # scrapes 3 hardcoded blog URLs, embeds, and populates the Pinecone index "langgraph-agentic-rag"
python main.py         # runs one example query through the compiled graph
pytest . -s -v         # run the full test suite (graph/tests/test_chains.py) — makes real API calls
```

Run a single test: `pytest graph/tests/test_chains.py::test_router_to_websearch -s -v`

### Architecture

The graph is assembled in `graph/graph.py` from nodes in `graph/nodes/` and chains (prompt+LLM+structured-output) in `graph/chains/`. State (`question`, `generation`, `web_search`, `documents`) is a single `TypedDict` in `graph/state.py`, passed through every node. Node/edge name constants live in `graph/consts.py` — use these instead of hardcoding strings when wiring the graph.

Control flow, as conditional edges in `graph.py`:
1. `route_question` (entry point) → `RETRIEVE` (vectorstore) or `WEBSEARCH`, decided by `chains/router.py`'s structured-output classifier.
2. `RETRIEVE` → `GRADE_DOCUMENTS`, which grades each doc's relevance (`chains/retrieval_grader.py`) and sets `web_search=True` if any doc is irrelevant.
3. `decide_to_generate` → `WEBSEARCH` (if any doc was graded irrelevant) or `GENERATE`.
4. `GENERATE` (`chains/generation.py`) → `grade_generation_grounded_in_documents_and_question`, which chains two graders: `hallucination_grader.py` (is the generation grounded in the docs?) then `answer_grader.py` (does it answer the question?) → routes to `END` ("useful"), back to `WEBSEARCH` ("not useful"), or back to `GENERATE` ("not supported"/hallucinated).

Compiling the graph (`app = workflow.compile()`) also regenerates `graph_output.png` via `draw_mermaid_png()` as a side effect of importing `graph.graph` — this requires network access to Mermaid's rendering service.

## Skills_and_prompts

Non-code documentation: Claude Code slash commands (`Commands/create-worktree.md`, `Commands/merge-worktree.md`) and reusable skill/prompt docs for FDE-style facilitation work (hackathon prep, stakeholder matrix, ADR writing). Edit these as markdown/frontmatter, not as application code — see `Skills_and_prompts/README.md` for a one-line description of each file's purpose.
