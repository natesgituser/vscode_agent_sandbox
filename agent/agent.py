#!/usr/bin/env python3
"""
Minimal LLM-capable code generation agent.

Usage:
  - Interactive: python agent/agent.py
  - Generate via CLI: python agent/agent.py generate mypkg --description "A sample package"

Behavior:
  - If environment variable OPENAI_API_KEY is set and --use-llm is provided, the agent will call OpenAI ChatCompletion to request code scaffolding.
  - Otherwise it uses a built-in template generator to write a minimal package skeleton.

This file is intentionally small and dependency-light.
"""

import os
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, List

try:
    import openai
except Exception:
    openai = None

TEMPLATE_INIT = """# Package init (generated)
"""

TEMPLATE_MAIN = """
"""

SETUP_PY = """
from setuptools import setup, find_packages

setup(
    name="{name}",
    version="0.0.1",
    packages=find_packages(),
    description="{desc}",
)
"""

README_PY = """
# {name}

{desc}
"""

TEST_SAMPLE = """
def test_import():
    import {name}
    assert True
"""


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")


def generate_local_package(dest: Path, name: str, desc: str):
    pkg_dir = dest / name
    # ensure the package exposes a convenient top-level API (hello)
    init_content = TEMPLATE_INIT + "\nfrom .main import hello\n__all__ = ['hello']\n"
    write_file(pkg_dir / "__init__.py", init_content)
    write_file(pkg_dir / "main.py", f"def hello():\n    return 'Hello from {name}'\n")
    write_file(dest / "setup.py", SETUP_PY.format(name=name, desc=desc))
    write_file(dest / "README.md", README_PY.format(name=name, desc=desc))
    write_file(dest / "tests" / f"test_{name}.py", TEST_SAMPLE.format(name=name))


def resolve_context7_library_id(library_name: str) -> Optional[str]:
    """Resolve a library name to a Context7-compatible library ID.

    This function expects two environment variables to be set when used:
      - CONTEXT7_API_URL: base URL of a Context7-compatible service (e.g. https://context7.example)
      - CONTEXT7_API_KEY: optional API key to authorize requests

    The implementation expects a POST endpoint at {API_URL}/resolve-library-id that
    accepts JSON {"libraryName": "..."} and returns JSON {"context7CompatibleLibraryID": "/org/project"}.

    This is intentionally flexible: if the environment is not configured the function
    returns None so the generator gracefully proceeds without extra context.
    """
    api_url = os.environ.get("CONTEXT7_API_URL")
    api_key = os.environ.get("CONTEXT7_API_KEY")
    if not api_url:
        return None
    url = api_url.rstrip("/") + "/resolve-library-id"
    body = json.dumps({"libraryName": library_name}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("context7CompatibleLibraryID")
    except Exception:
        return None


def get_context7_docs(library_id: str, topic: Optional[str] = None, tokens: int = 2000) -> Optional[str]:
    """Fetch documentation snippets for a resolved Context7 library ID.

    Expects POST {API_URL}/get-library-docs with JSON {"context7CompatibleLibraryID": id, "topic": topic, "tokens": tokens}
    returning {"docs": "..."}.
    Returns None on any error or if environment is not set.
    """
    api_url = os.environ.get("CONTEXT7_API_URL")
    api_key = os.environ.get("CONTEXT7_API_KEY")
    if not api_url or not library_id:
        return None
    url = api_url.rstrip("/") + "/get-library-docs"
    body = json.dumps({"context7CompatibleLibraryID": library_id, "topic": topic or "", "tokens": tokens}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("docs") or data.get("documentation") or data.get("content")
    except Exception:
        return None


def generate_with_openai(dest: Path, name: str, desc: str, model: str = "gpt-4", context7_enabled: bool = False, context7_libs: Optional[List[str]] = None):
    if not openai:
        raise RuntimeError("openai package not installed")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key

    # Build the prompt, optionally augmenting with Context7 docs if configured.
    prompt_parts = []
    prompt_parts.append(
        f"Create a minimal Python package named {name}. "
        f"Include __init__.py, main.py with a hello() function, a setup.py, README.md, and a single pytest test file. "
        f"Return a JSON object where keys are relative file paths and values are file contents. Short files only. Package description: {desc}"
    )

    if context7_enabled and context7_libs:
        for lib in context7_libs:
            lib = lib.strip()
            if not lib:
                continue
            lib_id = resolve_context7_library_id(lib)
            if lib_id:
                docs = get_context7_docs(lib_id)
                if docs:
                    # keep docs reasonably sized in the prompt
                    snippet = docs if len(docs) <= 4000 else docs[:4000] + "\n...[truncated]"
                    prompt_parts.append(f"\n--- Context7 docs for {lib} (id: {lib_id}) ---\n{snippet}\n")
            else:
                prompt_parts.append(f"\n--- Context7: could not resolve library {lib} ---\n")

    prompt = "\n".join(prompt_parts)

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200,
    )
    # attempt to parse content
    content = resp["choices"][0]["message"]["content"]
    # Expect JSON; try to find a JSON block
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError("OpenAI response did not contain JSON: " + content[:200])
    j = content[start:end+1]
    files = json.loads(j)
    for rel, text in files.items():
        write_file(dest / rel, text)


def main():
    p = argparse.ArgumentParser(prog="llm-agent")
    sub = p.add_subparsers(dest="cmd")

    gen = sub.add_parser("generate", help="Generate a python package")
    gen.add_argument("name", help="package name to create")
    gen.add_argument("--description", default="A small package", help="package description")
    gen.add_argument("--out", default=".", help="destination directory")
    gen.add_argument("--use-llm", action="store_true", help="use OpenAI to generate files (requires OPENAI_API_KEY)")
    gen.add_argument("--model", default="gpt-4", help="LLM model to use when --use-llm is set")
    gen.add_argument("--context7", action="store_true", help="Attempt to fetch Context7 docs and include them in the LLM prompt (requires CONTEXT7_API_URL)")
    gen.add_argument("--context7-libs", default="", help="Comma-separated list of library names to resolve via Context7 and include docs for")

    args = p.parse_args()

    if args.cmd is None:
        # interactive mode: ask for name and description
        name = input("Package name: ").strip()
        desc = input("Description: ").strip() or "A small package"
        out = Path(".")
        use_llm = input("Use OpenAI? [y/N]: ").lower().startswith("y")
        model = "gpt-4"
        if use_llm:
            print("Generating with OpenAI (requires OPENAI_API_KEY)")
            generate_with_openai(out, name, desc, model)
        else:
            generate_local_package(out, name, desc)
        return

    # CLI generate
    dest = Path(args.out)
    if args.use_llm:
        try:
            libs = [s.strip() for s in args.context7_libs.split(",") if s.strip()]
            generate_with_openai(dest, args.name, args.description, args.model, context7_enabled=args.context7, context7_libs=libs)
        except Exception as e:
            print("OpenAI generation failed:", e)
            print("Falling back to local template")
            generate_local_package(dest, args.name, args.description)
    else:
        generate_local_package(dest, args.name, args.description)


if __name__ == "__main__":
    main()
