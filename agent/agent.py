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

try:
    import openai
except Exception:
    openai = None

TEMPLATE_INIT = """"
"""  # empty init

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
    write_file(pkg_dir / "__init__.py", TEMPLATE_INIT)
    write_file(pkg_dir / "main.py", f"def hello():\n    return 'Hello from {name}'\n")
    write_file(dest / "setup.py", SETUP_PY.format(name=name, desc=desc))
    write_file(dest / "README.md", README_PY.format(name=name, desc=desc))
    write_file(dest / "tests" / f"test_{name}.py", TEST_SAMPLE.format(name=name))


def generate_with_openai(dest: Path, name: str, desc: str, model: str = "gpt-4"):
    if not openai:
        raise RuntimeError("openai package not installed")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key

    prompt = (
        f"Create a minimal Python package named {name}. "
        f"Include __init__.py, main.py with a hello() function, a setup.py, README.md, and a single pytest test file. "
        f"Return a JSON object where keys are relative file paths and values are file contents. "
        f"Short files only. Package description: {desc}"
    )

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
            generate_with_openai(dest, args.name, args.description, args.model)
        except Exception as e:
            print("OpenAI generation failed:", e)
            print("Falling back to local template")
            generate_local_package(dest, args.name, args.description)
    else:
        generate_local_package(dest, args.name, args.description)


if __name__ == "__main__":
    main()
