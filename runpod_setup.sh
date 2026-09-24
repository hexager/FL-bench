#!/usr/bin/env bash
# One-time setup for FL-bench on a RunPod GPU pod.
# Run this from the pod's terminal (SSH or the web terminal), not from a notebook cell.
#
# Before running, make sure GITHUB_TOKEN and WANDB_API_KEY are set — either as
# pod Environment Variables (set them when you deploy the pod) or exported here:
#   export GITHUB_TOKEN=xxxx
#   export WANDB_API_KEY=xxxx

set -euo pipefail

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN before running (pod env var or export)}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running (pod env var or export)}"

# Put the repo on the persistent volume if you attached one, so it survives pod restarts.
WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d "FL-bench" ]; then
  git clone "https://${GITHUB_TOKEN}@github.com/hexager/FL-bench.git"
fi
cd FL-bench
git pull

# tmux so long grid-search runs survive you closing the browser/SSH session
if ! command -v tmux &> /dev/null; then
  apt-get update -qq && apt-get install -y -qq tmux
fi

pip install -r .env/requirements.txt
pip install --no-cache-dir "wandb==0.28.0"

wandb login "$WANDB_API_KEY"

echo ""
echo "Setup done. Repo is at $WORKDIR/FL-bench"
echo ""
echo "Copy the patched grid_search.py and src/server/fedavg.py into the repo (or push them to your fork) first."
echo ""
echo "1) Calibrate on this pod (~10-15 min):"
echo "  python grid_search.py --methods fedavg fedprox elastic scaffold --calibrate"
echo ""
echo "2) Launch inside tmux so it survives disconnects:"
echo "  tmux new -s flbench"
echo "  python grid_search.py --methods fedavg fedprox elastic scaffold --workers 8 --gpus 0,1,2,3"
echo "  # detach: Ctrl+b then d   | reattach: tmux attach -t flbench"
