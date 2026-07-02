#!/bin/bash
# git_file_tree.sh
# Visualize all files in the git repo as a tree with file sizes.
#   RED    ◀  files > 100 MB  (LARGE FILE warning)
#   YELLOW    files  50–100 MB
#   dim       everything else
# Read-only: no staging, commits, or filesystem modifications.

cd "$(git rev-parse --show-toplevel)" || exit 1

if ! command -v python &>/dev/null; then
    echo "Error: python is required but not found." >&2
    exit 1
fi

# ── Python renderer (temp file avoids heredoc variable interpolation) ────────
TMPPY=$(mktemp /tmp/git_file_tree_XXXXXX.py)
trap "rm -f '$TMPPY'" EXIT

cat > "$TMPPY" << 'PYEOF'
import sys
import os

LARGE  = 100 * 1024 * 1024   # 100 MB  ── RED
MEDIUM =  50 * 1024 * 1024   #  50 MB  ── YELLOW

RED  = '\033[1;31m'
YEL  = '\033[1;33m'
BLU  = '\033[1;34m'
GRN  = '\033[0;32m'
BOLD = '\033[1m'
DIM  = '\033[2m'
NC   = '\033[0m'

# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_size(n):
    if n >= 1 << 30: return f'{n / (1 << 30):.2f} GB'
    if n >= 1 << 20: return f'{n / (1 << 20):.1f} MB'
    if n >= 1 << 10: return f'{n / (1 << 10):.1f} KB'
    return f'{n} B'

# ── read file list from stdin ─────────────────────────────────────────────────

paths = sorted(set(line.rstrip('\n') for line in sys.stdin if line.strip()))

file_sizes = {}   # path -> size in bytes
for p in paths:
    try:
        file_sizes[p] = os.path.getsize(p)
    except OSError:
        file_sizes[p] = 0

# ── build tree ────────────────────────────────────────────────────────────────
# Each leaf:  (kind='file', full_path)
# Each branch: (kind='dir',  children_dict)

def insert(tree, parts, full_path):
    name = parts[0]
    if len(parts) == 1:
        tree[name] = ('file', full_path)
    else:
        if name not in tree or tree[name][0] != 'dir':
            tree[name] = ('dir', {})
        insert(tree[name][1], parts[1:], full_path)

root = {}
for p in paths:
    parts = [c for c in p.replace('\\', '/').split('/') if c]
    if parts:
        insert(root, parts, p)

# ── render tree ───────────────────────────────────────────────────────────────

def render(node, prefix=''):
    # directories first, then files; each group sorted alphabetically
    entries = sorted(
        node.items(),
        key=lambda kv: (0 if kv[1][0] == 'dir' else 1, kv[0].lower())
    )
    for idx, (name, info) in enumerate(entries):
        is_last = idx == len(entries) - 1
        branch  = '└── ' if is_last else '├── '
        extend  = '    ' if is_last else '│   '
        kind, val = info

        if kind == 'dir':
            print(f'{prefix}{branch}{BLU}{BOLD}{name}/{NC}')
            render(val, prefix + extend)
        else:
            sz = file_sizes.get(val, 0)
            if sz >= LARGE:
                print(f'{prefix}{branch}'
                      f'{RED}{BOLD}{name}{NC}  '
                      f'{RED}[ {fmt_size(sz)} ]  ◀ LARGE FILE{NC}')
            elif sz >= MEDIUM:
                print(f'{prefix}{branch}'
                      f'{YEL}{name}{NC}  {YEL}[ {fmt_size(sz)} ]{NC}')
            else:
                print(f'{prefix}{branch}{name}  {DIM}[ {fmt_size(sz)} ]{NC}')

# ── print ─────────────────────────────────────────────────────────────────────

repo = os.path.basename(os.getcwd())
print()
print(f'{BOLD}{repo}/{NC}')
render(root)

# ── summary ───────────────────────────────────────────────────────────────────

total_sz  = sum(file_sizes.values())
large_lst = sorted(
    [(p, s) for p, s in file_sizes.items() if s >= LARGE],
    key=lambda x: -x[1]
)

print()
print(f'{DIM}{"─" * 62}{NC}')
print(f'{BOLD}Total:{NC}  {len(file_sizes)} files   {fmt_size(total_sz)}')

if large_lst:
    print()
    print(f'{RED}{BOLD}⚠  Files exceeding 100 MB  ({len(large_lst)} found):{NC}')
    for p, s in large_lst:
        print(f'  {RED}✗  {p}  [ {fmt_size(s)} ]{NC}')
    print()
    print(f'{YEL}  → Use git_lfs_push.sh to handle these files.{NC}')
else:
    print(f'{GRN}✓  No files exceed 100 MB{NC}')

print()
PYEOF

# ── collect tracked + untracked (non-ignored) files → pipe to renderer ───────
{
    git ls-files
    git ls-files --others --exclude-standard 2>/dev/null
} | sort -u | python "$TMPPY"