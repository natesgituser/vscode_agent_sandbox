# coding_server (simplified tools-only MVP)

This repository provides a minimal, offline-first toolkit of programmatic
tools for interacting with a codebase. The simplified package exposes
in-process tools (no persistence): filesystem helpers and a small tool
registry suitable for embedding in an agent loop. There is intentionally
no CLI.

Quick example — register a tool and use filesystem helpers:

```python
from time import sleep
from code_tools import (
    register_tool, list_tools, run_tool,
    search, atomic_write_text, read_text
)

# register a simple tool
def echo_tool(text: str) -> str:
    return 'ECHO: ' + text

register_tool('echo', echo_tool)

# agent loop (toy example)
for i in range(3):
    out = run_tool('echo', f'iteration {i}')
    print('tool result:', out)

    # search project for TODOs (non-regex search)
    hits = search('.', 'TODO', use_regex=False, max_results=10)
    print('hits:', len(hits))
    sleep(0.1)

# atomic file write/read example
atomic_write_text('example.txt', 'line1\nline2\n')
print(read_text('example.txt'))
```

Validation — run these commands from the project root (`/home/vuser/Desktop/tool_test`):

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, '.')
from code_tools import (
    register_tool, run_tool, atomic_write_text, read_text
)

def echo(x):
    return f'ECHO:{x}'

register_tool('echo', echo)
print(run_tool('echo', 'hello'))
atomic_write_text('example.txt', 'one\ntwo\n')
print(read_text('example.txt'))
PY
```

Notes:
- `exec_shell` is provided but runs arbitrary commands — only use with trusted inputs.
- This simplified package keeps everything in-process and dependency-free.
