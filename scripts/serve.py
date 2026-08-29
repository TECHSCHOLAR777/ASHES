#!/usr/bin/env python3
"""Serve the ASHES copilot UI: python scripts/serve.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.serve.app:app", host="0.0.0.0", port=int(__import__("os").environ.get("ASHES_UI_PORT", "8080")), reload=False)
