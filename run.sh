#!/usr/bin/env bash
# One-command runner: creates a venv if needed, installs deps, runs the pipeline.
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is not set."
    echo "Get a free key at https://aistudio.google.com and either:"
    echo "  export GEMINI_API_KEY=your_key_here"
    echo "  or copy .env.example to .env and fill it in"
    exit 1
fi

if ! curl -s -o /dev/null -w '' "http://localhost:11434/api/tags" 2>/dev/null; then
    echo "WARNING: Ollama doesn't seem to be running on localhost:11434."
    echo "Start it with 'ollama serve' and make sure a vision model is pulled:"
    echo "  ollama pull llava"
fi

python main.py --images images/ --mode both "$@"
