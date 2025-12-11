Minimal LLM code-generation agent

This folder contains a tiny agent that can scaffold a Python package.

Quickstart

1. (Optional) create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate a package interactively:

```bash
python agent/agent.py
```

3. Generate via CLI (local template):

```bash
python agent/agent.py generate samplepkg --description "A sample package" --out ./output
```

4. Generate via OpenAI (requires OPENAI_API_KEY):

```bash
export OPENAI_API_KEY="sk-..."
python agent/agent.py generate samplepkg --description "Use OpenAI to scaffold" --use-llm --model gpt-4
```

Notes

- If OpenAI generation fails, the agent falls back to the simple local template generator.
- The generated layout is intentionally minimal: package dir with `__init__.py` and `main.py`, `setup.py`, `README.md`, and a test.

MCP example

This repository includes a tiny MCP-style example tool at `agent/mcp_server.py`.

Usage examples:

Run a single test (no server):

```bash
python agent/mcp_server.py --test '{"name":"samplepkg","description":"from mcp example","out":"./tmp_output"}'
```

Start a small HTTP example server (POST JSON to /mcp):

```bash
python agent/mcp_server.py --serve --port 8000
# then POST JSON like {"name":"pkg","description":"d","out":"./tmp_output"} to http://localhost:8000/mcp
```
