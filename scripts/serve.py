#!/usr/bin/env python3
"""Serve the ASHES copilot UI: python scripts/serve.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.serve.app:app", host="0.0.0.0", port=int(__import__("os").environ.get("ASHES_UI_PORT", "8080")), reload=False)
