"""Combined code tools at repository root.

Moved out of `coding_server/` to reduce nesting per user request.
"""
from typing import Any, List, Optional, Dict, Callable
import os
import re
import subprocess
import json

# Tool registry
_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_tool(name: str, func: Callable[..., Any]) -> None:
    """Register a callable under `name`. Overwrites existing registration."""
    _REGISTRY[name] = func


def list_tools() -> List[str]:
    return list(_REGISTRY.keys())


def run_tool(name: str, *args, **kwargs) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"tool not registered: {name}")
    return _REGISTRY[name](*args, **kwargs)


# Filesystem operations
def read_text(path: str, encoding: str = 'utf-8') -> str:
    with open(path, 'r', encoding=encoding) as f:
        return f.read()


def atomic_write_text(path: str, content: str, encoding: str = 'utf-8') -> None:
    """Simplified write: create parent dirs and write directly (no temp file)."""
    d = os.path.dirname(path) or '.'
    os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding=encoding) as f:
        f.write(content)


def safe_mkdir(path: str, exist_ok: bool = True) -> None:
    os.makedirs(path, exist_ok=exist_ok)


def list_dir(path: str = '.', recursive: bool = False, pattern: Optional[str] = None) -> List[str]:
    out = []
    if recursive:
        for dirpath, _, files in os.walk(path):
            for fn in files:
                p = os.path.join(dirpath, fn)
                if pattern is None or pattern in fn:
                    out.append(p)
    else:
        for fn in os.listdir(path):
            p = os.path.join(path, fn)
            if pattern is None or pattern in fn:
                out.append(p)
    return out


def search(root: str, pattern: str, use_regex: bool = True, max_results: int = 1000) -> List[Dict[str, Any]]:
    rx = re.compile(pattern) if use_regex else None
    results = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            path = os.path.join(dirpath, fn)
            try:
                with open(path, 'r', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if (rx and rx.search(line)) or (not rx and pattern in line):
                            results.append({'path': path, 'line_no': i, 'line': line.rstrip('\n')})
                            if len(results) >= max_results:
                                return results
            except Exception:
                continue
    return results


def delete_lines(path: str, start_line: int, end_line: int) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start = max(1, start_line) - 1
    end = max(0, end_line)
    new = lines[:start] + lines[end:]
    atomic_write_text(path, ''.join(new))


def replace_lines(path: str, start_line: int, end_line: int, new_lines: List[str] or str) -> None:
    if isinstance(new_lines, str):
        new_lines = [new_lines]
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start = max(1, start_line) - 1
    end = max(0, end_line)
    new = lines[:start] + [ln if ln.endswith('\n') else ln + '\n' for ln in new_lines] + lines[end:]
    atomic_write_text(path, ''.join(new))


def insert_at_line(path: str, lineno: int, text: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    idx = max(0, lineno - 1)
    if not text.endswith('\n'):
        text = text + '\n'
    new = lines[:idx] + [text] + lines[idx:]
    atomic_write_text(path, ''.join(new))


def exec_shell(cmd: str, timeout: Optional[float] = None, capture_output: bool = True, allow_fails: bool = False) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, timeout=timeout)
        result = {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
        if proc.returncode != 0 and not allow_fails:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
        return result
    except subprocess.CalledProcessError as e:
        if allow_fails:
            return {'returncode': e.returncode, 'stdout': getattr(e, 'output', ''), 'stderr': getattr(e, 'stderr', '')}
        raise


# Simple import/export helpers (operate on JSON snapshots of registered tool metadata and files)
def export_store(path: str) -> None:
    """Export a minimal snapshot: list of registered tool names and a file list.

    Note: since memory persistence was removed, export_store produces a small
    JSON with the current registry and optionally files of interest.
    """
    snapshot = {'tools': list(_REGISTRY.keys())}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)


def import_store(path: str, merge: bool = True) -> None:
    """Import a snapshot produced by `export_store`. This will not restore
    runtime callables; it only updates registry metadata (no-op for now).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # currently we don't recreate callables from disk; future implementations
    # may load plugin modules or similar. For now, just return the parsed JSON.
    return data
