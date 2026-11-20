"""Prepare a reproducible, deduplicated UTF-8 corpus from licensed source files."""
import argparse
import hashlib
import json
from pathlib import Path

EXTENSIONS = {'.py','.js','.ts','.tsx','.jsx','.html','.css','.lua','.luau','.kt','.java','.c','.h','.cpp','.hpp','.cs','.rs','.go','.php','.rb','.sql','.sh','.swift','.dart','.r','.scala','.ex','.exs','.pl','.jl','.hs','.fs','.fsx','.clj','.cljs','.elm','.vue','.svelte','.json','.md','.txt','.yaml','.yml','.toml','.xml','.zig','.nim','.pas','.asm','.f90','.m','.mm','.gradle','.kts','.bat','.ps1'}
EXCLUDED = {'.git','node_modules','build','dist','vendor','.gradle','__pycache__','.venv','venv','target'}

def build(root: Path, output: Path, max_bytes: int=200_000):
    if not root.is_dir(): raise ValueError('Input must be an existing directory')
    if max_bytes < 128: raise ValueError('max-bytes must be >= 128')
    seen=set(); chunks=[]; count=0; skipped=0
    for file in sorted(root.rglob('*')):
        if not file.is_file() or any(part in EXCLUDED for part in file.relative_to(root).parts) or file.suffix.lower() not in EXTENSIONS:
            continue
        if file.is_symlink() or file.stat().st_size > max_bytes:
            skipped+=1; continue
        raw=file.read_bytes()
        if b'\x00' in raw: skipped+=1; continue
        try: text=raw.decode('utf-8')
        except UnicodeDecodeError: skipped+=1; continue
        if not text.strip(): continue
        digest=hashlib.sha256(raw).hexdigest()
        if digest in seen: continue
        seen.add(digest); count+=1
        chunks.append(f'\n<file language="{file.suffix.lstrip(".")}">\n{text.rstrip()}\n</file>\n')
    if not chunks: raise ValueError('No eligible text files found')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(chunks),encoding='utf-8')
    return {'files':count,'skipped':skipped,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--max-bytes',type=int,default=200000)
    a=ap.parse_args(); print(json.dumps(build(a.input,a.output,a.max_bytes),indent=2))
