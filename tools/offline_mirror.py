#!/usr/bin/env python3
"""
Offline mirror helper

Downloads packages and Context7 docs to a cache directory while reporting
per-file progress, elapsed time, ETA, and current cache size. Designed to be
safe for interactive use: use `--dry-run` to estimate sizes without downloading.

Usage examples:
  python tools/offline_mirror.py --pip requirements.txt --npm npm.txt --apt apt.txt --context7 context7_libs.txt --cache ./cache

Notes:
- This script uses the PyPI and npm registries HTTP APIs to estimate sizes
  when possible and then streams downloads showing progress.
- For apt packages it attempts to use `apt-get --print-uris` to obtain .deb URIs
  and sizes; this requires `apt-get` and network access.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

try:
    # use urllib from stdlib to avoid extra deps
    from urllib.request import urlopen, Request
    from urllib.parse import quote
except Exception:
    print("urllib not available; this script requires Python stdlib urllib")
    raise


CHUNK = 64 * 1024


def human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for u in ["KiB", "MiB", "GiB", "TiB"]:
        n /= 1024.0
        if n < 1024:
            return f"{n:.2f} {u}"
    return f"{n:.2f} PiB"


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def download_stream(url: str, dest: Path, expected: Optional[int] = None, dry_run: bool = False) -> Tuple[int, float]:
    """
    Download a URL to dest while reporting progress. Returns (bytes_written, seconds_elapsed).
    If dry_run True, does not write and only tries to estimate size via HEAD.
    """
    start = time.time()
    headers = {"User-Agent": "offline-mirror/1.0"}
    req = Request(url, headers=headers)
    if dry_run:
        # try HEAD via Request with method='HEAD' when supported
        try:
            req.get_method = lambda: 'HEAD'  # type: ignore[attr-defined]
            with urlopen(req, timeout=15) as resp:
                size = resp.getheader('Content-Length')
                return (int(size) if size else 0, 0.0)
        except Exception:
            return (expected or 0, 0.0)

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with urlopen(req, timeout=60) as resp, open(dest, 'wb') as fd:
            # try to get content-length
            length = resp.getheader('Content-Length')
            total = int(length) if length and length.isdigit() else expected
            last_report = start
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fd.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_report >= 0.5:
                    elapsed = now - start
                    speed = written / elapsed if elapsed > 0 else 0
                    perc = (written / total * 100) if total else 0
                    eta = (total - written) / speed if speed and total and written < total else 0
                    print(f"  {dest.name}: {human(written)} / {human(total or 0)} ({perc:.1f}%) speed={human(int(speed))}/s eta={eta:.1f}s")
                    last_report = now
    except Exception as e:
        print("Download failed:", e)
        # remove partial file if exists
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        raise
    elapsed = time.time() - start
    return written, elapsed


def fetch_pypi(pkg: str, cache: Path, dry_run: bool = False) -> Tuple[int, str]:
    # pkg may be 'name' or 'name==version'
    if '==' in pkg:
        name, version = pkg.split('==', 1)
    else:
        name, version = pkg, None
    api = f"https://pypi.org/pypi/{quote(name)}/json"
    print(f"Resolving PyPI package {pkg} -> {api}")
    try:
        with urlopen(api, timeout=15) as resp:
            data = json.load(resp)
    except Exception as e:
        print("  Failed to query PyPI:", e)
        return 0, ""
    releases = data.get('releases', {})
    target = None
    if version:
        target = releases.get(version)
    else:
        # choose latest release info from 'info.version'
        version = data.get('info', {}).get('version')
        target = releases.get(version)
    if not target:
        print("  Could not find release metadata for", pkg)
        return 0, ""
    # choose a wheel first then sdist
    file_info = None
    for f in target:
        if f.get('packagetype') in ('bdist_wheel', 'bdist_egg'):
            file_info = f
            break
    if not file_info and target:
        file_info = target[0]
    if not file_info:
        print("  No files discovered for", pkg)
        return 0, ""
    url = file_info['url']
    size = file_info.get('size')
    fname = Path(url.split('/')[-1])
    dest = cache / 'pypi' / fname
    print(f"  Downloading PyPI {pkg} -> {dest} (estimated {human(size or 0)})")
    if dry_run:
        return (size or 0), str(dest)
    b, elapsed = download_stream(url, dest, expected=size)
    return b, str(dest)


def fetch_npm(pkg: str, cache: Path, dry_run: bool = False) -> Tuple[int, str]:
    # pkg may be scoped like @upstash/context7-mcp or with @version
    # split version suffix if present like pkg@1.2.3
    version = None
    if '@' in pkg and not pkg.startswith('@'):
        # name@version
        name, ver = pkg.rsplit('@', 1)
        pkg = name
        version = ver

    api_name = pkg
    # npm registry needs scoped packages encoded with %2f for '/'
    api_url = f"https://registry.npmjs.org/{quote(api_name, safe='')}"
    print(f"Resolving npm package {pkg} -> {api_url}")
    try:
        with urlopen(api_url, timeout=15) as resp:
            data = json.load(resp)
    except Exception as e:
        print("  Failed to query npm registry:", e)
        return 0, ""
    if version is None:
        version = data.get('dist-tags', {}).get('latest')
    ver_info = data.get('versions', {}).get(version)
    if not ver_info:
        print("  Could not find version info for", pkg)
        return 0, ""
    tarball = ver_info.get('dist', {}).get('tarball')
    if not tarball:
        print("  No tarball URL for", pkg)
        return 0, ""
    fname = Path(tarball.split('/')[-1])
    dest = cache / 'npm' / fname
    print(f"  Downloading npm {pkg}@{version} -> {dest}")
    if dry_run:
        # try HEAD to estimate size
        try:
            req = Request(tarball, headers={"User-Agent": "offline-mirror/1.0"})
            req.get_method = lambda: 'HEAD'  # type: ignore[attr-defined]
            with urlopen(req, timeout=15) as r:
                size = r.getheader('Content-Length')
                return (int(size) if size else 0), str(dest)
        except Exception:
            return 0, str(dest)
    b, elapsed = download_stream(tarball, dest, expected=None)
    return b, str(dest)


def apt_get_print_uris(pkg: str) -> List[Tuple[str, int]]:
    """Run apt-get --print-uris and parse quoted URIs and size in bytes.
    Returns list of (uri, size).
    """
    cmd = ["apt-get", "-qq", "--print-uris", "install", "--yes", "--download-only", pkg]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        out = e.output
    uris = []
    for line in out.splitlines():
        # lines often contain: 'http://.../package.deb' package_version size md5
        if "http" in line and "'" in line:
            # extract quoted URL
            try:
                first = line.index("'")
                second = line.index("'", first + 1)
                url = line[first + 1:second]
                parts = line[second + 1:].strip().split()
                # try to find numeric size in parts
                size = None
                for p in parts[::-1]:
                    if p.isdigit():
                        size = int(p)
                        break
                uris.append((url, size or 0))
            except Exception:
                continue
    return uris


def fetch_apt(pkg: str, cache: Path, dry_run: bool = False) -> Tuple[int, List[str]]:
    entries = apt_get_print_uris(pkg)
    if not entries:
        print(f"  No URIs found for apt package {pkg} (apt may be unavailable)")
        return 0, []
    downloaded = 0
    dests = []
    for url, size in entries:
        fname = Path(url.split('/')[-1])
        dest = cache / 'apt' / fname
        print(f"  apt: {fname} size={human(size)} url={url}")
        if dry_run:
            downloaded += size or 0
            dests.append(str(dest))
            continue
        b, _ = download_stream(url, dest, expected=size)
        downloaded += b
        dests.append(str(dest))
    return downloaded, dests


def fetch_context7(lib: str, cache: Path, api_url: Optional[str], api_key: Optional[str], dry_run: bool = False) -> Tuple[int, str]:
    if not api_url:
        print("  CONTEXT7_API_URL not provided; skipping", lib)
        return 0, ""
    url = api_url.rstrip('/') + '/get-library-docs'
    payload = {"context7CompatibleLibraryID": lib, "topic": "", "tokens": 4000}
    data = json.dumps(payload).encode('utf-8')
    headers = {"Content-Type": "application/json", "User-Agent": "offline-mirror/1.0"}
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    req = Request(url, data=data, headers=headers)
    print(f"  Fetching Context7 docs for {lib} -> {url}")
    if dry_run:
        return 0, ""
    try:
        with urlopen(req, timeout=30) as resp:
            docs = json.load(resp).get('docs') or json.load(resp)
    except Exception as e:
        print("  Failed to fetch docs:", e)
        try:
            # try reading raw text
            with urlopen(req, timeout=30) as resp:
                docs = resp.read().decode('utf-8')
        except Exception:
            return 0, ""
    if not docs:
        return 0, ""
    fname = f"context7_{lib.strip('/').replace('/', '_')}.md"
    dest = cache / 'context7' / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(docs, encoding='utf-8')
    return dest.stat().st_size, str(dest)


def main():
    p = argparse.ArgumentParser(description="Offline mirror and progress reporter for apt/pip/npm/Context7 docs")
    p.add_argument('--cache', default='./cache', help='Cache directory')
    p.add_argument('--pip', help='Path to pip requirements.txt')
    p.add_argument('--npm', help='Path to newline-separated npm package names (pkg or pkg@ver)')
    p.add_argument('--apt', help='Path to newline-separated apt package names')
    p.add_argument('--context7', help='File with context7 library IDs or names, one per line')
    p.add_argument('--dry-run', action='store_true', help='Do not download, only estimate sizes')
    p.add_argument('--confirm', action='store_true', help='Automatically confirm downloads')
    args = p.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    total_estimated = 0
    work_items = []  # tuples (kind, spec)

    if args.pip:
        with open(args.pip, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                work_items.append(('pypi', s))

    if args.npm:
        with open(args.npm, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                work_items.append(('npm', s))

    if args.apt:
        with open(args.apt, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                work_items.append(('apt', s))

    if args.context7:
        with open(args.context7, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                work_items.append(('context7', s))

    print(f"Planning to process {len(work_items)} items into cache {cache}")

    # Quick estimate pass (dry-run style) to get expected total size where possible
    for kind, spec in work_items:
        try:
            if kind == 'pypi':
                b, _ = fetch_pypi(spec, cache, dry_run=True)
                total_estimated += b or 0
            elif kind == 'npm':
                b, _ = fetch_npm(spec, cache, dry_run=True)
                total_estimated += b or 0
            elif kind == 'apt':
                entries = apt_get_print_uris(spec)
                for _, size in entries:
                    total_estimated += size or 0
            elif kind == 'context7':
                # unknown size; skip
                pass
        except Exception as e:
            print("  estimate error for", spec, e)

    print(f"Estimated total download size: {human(total_estimated)}")
    if not args.confirm and not args.dry_run:
        ok = input('Proceed with downloads? [y/N]: ').lower().startswith('y')
        if not ok:
            print('Aborting')
            return

    downloaded_total = 0
    api_url = os.environ.get('CONTEXT7_API_URL')
    api_key = os.environ.get('CONTEXT7_API_KEY')

    for idx, (kind, spec) in enumerate(work_items, start=1):
        print(f"[{idx}/{len(work_items)}] {kind}: {spec}")
        start = time.time()
        try:
            if kind == 'pypi':
                b, path = fetch_pypi(spec, cache, dry_run=args.dry_run)
                downloaded_total += b or 0
            elif kind == 'npm':
                b, path = fetch_npm(spec, cache, dry_run=args.dry_run)
                downloaded_total += b or 0
            elif kind == 'apt':
                b, paths = fetch_apt(spec, cache, dry_run=args.dry_run)
                downloaded_total += b or 0
            elif kind == 'context7':
                b, path = fetch_context7(spec, cache, api_url, api_key, dry_run=args.dry_run)
                downloaded_total += b or 0
        except Exception as e:
            print("  Error processing item:", e)
            continue
        elapsed = time.time() - start
        remaining = max(0, total_estimated - downloaded_total)
        size_now = dir_size(cache)
        print(f"  Completed in {elapsed:.1f}s; downloaded so far: {human(downloaded_total)}; remaining (est): {human(remaining)}; cache size: {human(size_now)}")

    print("All items processed.")
    final_size = dir_size(cache)
    print(f"Final cache size: {human(final_size)}")


if __name__ == '__main__':
    main()
