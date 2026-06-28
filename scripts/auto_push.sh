#!/bin/bash
# Auto-push script for energivanu2
# Usage: bash scripts/auto_push.sh "commit message"

cd /home/work/.openclaw/workspace/energivanu2

MSG="${1:-Auto-update: $(date +%Y-%m-%d_%H:%M)}"

git add -A
if git diff --cached --quiet; then
    echo "Nothing to commit"
    exit 0
fi

git commit -m "$MSG" && git push origin main 2>&1
echo "✅ Pushed: $MSG"
