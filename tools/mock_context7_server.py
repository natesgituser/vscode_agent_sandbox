#!/usr/bin/env python3
"""
Simple mock Context7 API server for demonstration.

Endpoints (POST JSON):
- /resolve-library-id {"libraryName":"scapy"} -> {"context7CompatibleLibraryID":"/pypi/scapy"}
- /get-library-docs {"context7CompatibleLibraryID":"/pypi/scapy"} -> {"docs":"..."}

This server reads from the `cache` directory created by `tools/offline_mirror.py`:
- PyPI wheels: cache/pypi/*.whl -> returns a short metadata string
- Man pages: cache/man/ip.8.html -> returns its HTML as docs

Run: python3 tools/mock_context7_server.py --port 9001
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from typing import Optional
import argparse
import os
import re
from urllib.request import urlopen, Request
from urllib.error import URLError
import gzip
from io import BytesIO

CACHE = Path('cache')
PIP_MIRROR = os.environ.get('PIP_MIRROR_URL')  # e.g. http://localhost:8080/simple
MAN_MIRROR = os.environ.get('MAN_MIRROR_URL')  # e.g. http://localhost:8000/man


def read_maybe_gz(path: Path) -> str:
    try:
        if path.suffix == '.gz':
            with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''


def find_system_man(name: str) -> Optional[Path]:
    # Common man locations and sections to try
    locations = [Path('/usr/share/man'), Path('/usr/local/share/man')]
    sections = ['8', '1', '7', '3']
    for base in locations:
        for sec in sections:
            p = base / f'man{sec}' / f'{name}.{sec}'
            if p.exists():
                return p
            # gzipped variant
            pgz = Path(str(p) + '.gz')
            if pgz.exists():
                return pgz
    # not found
    return None


def find_local_pip_file(pkg: str) -> Optional[Path]:
    # Look into PIP_CACHE_DIR, default ~/.cache/pip/wheels
    pip_cache = os.environ.get('PIP_CACHE_DIR')
    if pip_cache:
        base = Path(pip_cache)
    else:
        base = Path.home() / '.cache' / 'pip' / 'wheels'
    if base.exists():
        # search for files that start with pkg name
        for p in base.rglob(f'{pkg}*'):
            if p.is_file():
                return p
    # also check common local wheelhouse
    local_wheelhouse = Path('wheelhouse')
    if local_wheelhouse.exists():
        for p in local_wheelhouse.rglob(f'{pkg}*'):
            if p.is_file():
                return p
    return None


def safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        return ''


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        b = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length) if length else b''
        try:
            payload = json.loads(raw.decode('utf-8') or '{}')
        except Exception:
            payload = {}

        if self.path.endswith('/resolve-library-id'):
            lib = payload.get('libraryName', '')
            # map simple names -> our cached ids
            if lib.lower() == 'scapy' or 'scapy' in lib:
                res = {'context7CompatibleLibraryID': '/pypi/scapy'}
            elif lib.lower() in ('ip', 'ip(8)') or 'ip' in lib:
                res = {'context7CompatibleLibraryID': '/man/ip'}
            else:
                res = {'context7CompatibleLibraryID': lib}
            self._send_json(res)
            return

        if self.path.endswith('/get-library-docs'):
            libid = payload.get('context7CompatibleLibraryID') or payload.get('context7compatiblelibraryid')
            if not libid:
                return self._send_json({'error': 'missing id'}, status=400)
            # For PyPI ids try to prefer a configured PIP_MIRROR; otherwise fall back to cache
            if libid.startswith('/pypi/'):
                pkg = libid.split('/')[-1]
                # try local PIP mirror first
                if PIP_MIRROR:
                    try:
                        simple_url = PIP_MIRROR.rstrip('/') + f"/{pkg}/"
                        req = Request(simple_url, headers={'User-Agent': 'mock-context7/1.0'})
                        with urlopen(req, timeout=10) as resp:
                            html = resp.read().decode('utf-8', errors='ignore')
                        # find the first wheel or sdist link
                        m = re.search(r'href\s*=\s*"([^"]+\.(whl|tar.gz|zip))"', html, re.IGNORECASE)
                        if m:
                            file_url = m.group(1)
                            # if link is relative, join with simple_url
                            if not file_url.startswith('http'):
                                if file_url.startswith('/'):
                                    base = PIP_MIRROR.rstrip('/')
                                    file_url = base + file_url
                                else:
                                    file_url = simple_url + file_url
                            # try HEAD to get size
                            try:
                                head = Request(file_url, method='HEAD', headers={'User-Agent': 'mock-context7/1.0'})
                                with urlopen(head, timeout=10) as hresp:
                                    size = hresp.getheader('Content-Length')
                                    size = int(size) if size and size.isdigit() else None
                            except Exception:
                                size = None
                            docs = f"PyPI mirror entry: {file_url}\nsize={size or 'unknown'} bytes"
                            return self._send_json({'docs': docs})
                    except URLError as e:
                        # mirror unreachable; fall back to cache
                        pass
                # fallback: use cache/pypi
                p = next((CACHE / 'pypi').glob(f'{pkg}*'), None)
                if p:
                    docs = f"PyPI cached file: {p.name}\nsize={p.stat().st_size} bytes"
                else:
                    docs = "PyPI package not found in cache or mirror"
                return self._send_json({'docs': docs})

            # For man pages try MAN_MIRROR first
            if libid.startswith('/man/'):
                name = libid.split('/')[-1]
                if MAN_MIRROR:
                    try:
                        man_url = MAN_MIRROR.rstrip('/') + f'/{name}.html'
                        req = Request(man_url, headers={'User-Agent': 'mock-context7/1.0'})
                        with urlopen(req, timeout=10) as resp:
                            html = resp.read().decode('utf-8', errors='ignore')
                        return self._send_json({'docs': html})
                    except URLError:
                        pass
                # fallback to cache
                p = CACHE / 'man' / f'{name}.html'
                if p.exists():
                    html = safe_read_text(p)
                    return self._send_json({'docs': html})
                else:
                    return self._send_json({'docs': 'manpage not found in cache or mirror'}, status=404)

            # default: echo
            return self._send_json({'docs': f'No docs for {libid} (mocked server)'} )

        # unknown endpoint
        self._send_json({'error': 'unknown endpoint'}, status=404)

    def log_message(self, format, *args):
        # keep output concise
        print(format % args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=9001)
    args = p.parse_args()
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    print(f"Mock Context7 server listening on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Shutting down')
        server.server_close()


if __name__ == '__main__':
    main()
