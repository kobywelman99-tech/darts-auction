#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  python src/build_history.py   # builds the dataset (cached, fast on rerun)"
echo "  python src/scarcity.py        # replacement level + implied pricing"
