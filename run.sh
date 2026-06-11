#!/usr/bin/env bash
# Generate the campaign data and compare uplift meta-learners.
set -e
cd "$(dirname "$0")"

pip install -r requirements.txt
python3 src/generate_data.py
python3 src/evaluate.py
echo ""
echo "See reports/qini_curves.png and reports/results.json"
