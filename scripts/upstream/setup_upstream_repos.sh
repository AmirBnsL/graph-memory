#!/usr/bin/env bash
#
# Idempotently clone the three official upstream conversational memory repositories
# into external/ (gitignored) and record exact pinned commit SHAs in provenance/upstream_repos.json.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL_DIR="$ROOT/external"
PROVENANCE_FILE="$ROOT/provenance/upstream_repos.json"

mkdir -p "$EXTERNAL_DIR"
mkdir -p "$ROOT/provenance"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

# 1. MAGMA (ACL 2026)
MAGMA_DIR="$EXTERNAL_DIR/MAGMA"
MAGMA_REPO="https://github.com/FredJiang0324/MAGMA.git"
if [ ! -d "$MAGMA_DIR/.git" ]; then
    say "Cloning MAGMA from $MAGMA_REPO"
    git clone --quiet "$MAGMA_REPO" "$MAGMA_DIR"
else
    say "MAGMA already cloned"
fi
MAGMA_SHA="$(git -C "$MAGMA_DIR" rev-parse HEAD)"

# 2. EMem (UIUC 2025)
EMEM_DIR="$EXTERNAL_DIR/EMem"
EMEM_REPO="https://github.com/KevinSRR/EMem.git"
if [ ! -d "$EMEM_DIR/.git" ]; then
    say "Cloning EMem from $EMEM_REPO"
    git clone --quiet "$EMEM_REPO" "$EMEM_DIR"
else
    say "EMem already cloned"
fi
EMEM_SHA="$(git -C "$EMEM_DIR" rev-parse HEAD)"

# 3. AgenticMemory / A-MEM (NeurIPS 2025)
AMEM_DIR="$EXTERNAL_DIR/AgenticMemory"
AMEM_REPO="https://github.com/WujiangXu/AgenticMemory.git"
if [ ! -d "$AMEM_DIR/.git" ]; then
    say "Cloning AgenticMemory from $AMEM_REPO"
    git clone --quiet "$AMEM_REPO" "$AMEM_DIR"
else
    say "AgenticMemory already cloned"
fi
AMEM_SHA="$(git -C "$AMEM_DIR" rev-parse HEAD)"

# Write provenance record
python3 -c "
import json
prov = {
    'repositories': {
        'MAGMA': {
            'repo': '$MAGMA_REPO',
            'commit': '$MAGMA_SHA',
            'venue': 'ACL 2026',
            'paper': 'MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents (arXiv:2601.03236)'
        },
        'EMem': {
            'repo': '$EMEM_REPO',
            'commit': '$EMEM_SHA',
            'venue': 'Preprint 2025',
            'paper': 'A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents (arXiv:2511.17208)'
        },
        'AgenticMemory': {
            'repo': '$AMEM_REPO',
            'commit': '$AMEM_SHA',
            'venue': 'NeurIPS 2025',
            'paper': 'A-MEM: Agentic Memory for LLM Agents (arXiv:2502.12110)'
        }
    }
}
with open('$PROVENANCE_FILE', 'w') as f:
    json.dump(prov, f, indent=2)
"

say "Upstream repos pinned:"
say "  MAGMA: $MAGMA_SHA"
say "  EMem: $EMEM_SHA"
say "  A-MEM: $AMEM_SHA"
say "Provenance written to $PROVENANCE_FILE"
