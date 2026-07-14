#!/usr/bin/env python3
"""CLI wrapper kept stable for contest packaging."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinical_nlp_pipeline import main

if __name__ == "__main__":
    main()
