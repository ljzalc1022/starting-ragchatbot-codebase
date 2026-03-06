# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the server (from repo root)
./run.sh

# Or manually (must run from backend/ directory)
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app is served at `http://localhost:8000`. FastAPI auto-docs at `http://localhost:8000/docs`.

The server **must be started from the `backend/` directory** — all imports are relative and `../docs` / `../frontend` paths depend on this.

## Architecture

**Full-stack RAG chatbot** using Anthropic tool use to decide when to search a ChromaDB vector store.

### Backend (`backend/`)

All modules live flat in `backend/` and import each other directly (no packages).

| File | Role |
|---|---|
| `app.py` | FastAPI app. Two API routes (`POST /api/query`, `GET /api/courses`) + serves `../frontend` as static files. Loads `../docs` into the vector store on startup. |
| `rag_system.py` | Central orchestrator. Wires together all components and exposes `query()` and `add_course_folder()`. |
| `ai_generator.py` | Wraps the Anthropic SDK. Makes two Claude API calls when tool use is triggered: one to get the tool call, one to get the final answer after injecting tool results. |
| `vector_store.py` | ChromaDB wrapper with two collections: `course_catalog` (course-level metadata) and `course_content` (text chunks). Course names are resolved semantically before filtering. |
| `search_tools.py` | `Tool` ABC + `CourseSearchTool` implementation + `ToolManager` registry. Tools expose an Anthropic-compatible JSON schema via `get_tool_definition()`. |
| `document_processor.py` | Parses `.txt` course files into `Course` + `CourseChunk` objects, then splits content into overlapping sentence-based chunks. |
| `session_manager.py` | In-memory conversation history (not persisted across restarts). History is injected as plain text into the Claude system prompt. |
| `config.py` | Single `Config` dataclass loaded from `.env`. Key params: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, `MAX_RESULTS=5`, `MAX_HISTORY=2`. |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk`. |

### Tool Use Flow

Claude is given the `search_course_content` tool on every query but decides autonomously whether to use it. When it does:
1. `ToolManager.execute_tool()` dispatches to `CourseSearchTool.execute()`
2. `VectorStore.search()` optionally resolves a fuzzy course name via the `course_catalog` collection, then queries `course_content` with an optional `where` filter
3. Results are injected back into Claude for a second API call

### Document Format

Course `.txt` files in `docs/` must follow this structure:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <title>
...
```
Course titles act as unique IDs in ChromaDB. Re-running the server skips already-indexed courses.

### Frontend (`frontend/`)

Vanilla JS — no build step. `script.js` POSTs to `/api/query` with `{query, session_id}` and renders the response using `marked.parse()` for markdown. Session ID is stored in a JS variable (`currentSessionId`) and sent with every subsequent request.
