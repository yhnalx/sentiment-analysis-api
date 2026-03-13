#!/usr/bin/env bash
set -e

python download_model.py
uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"