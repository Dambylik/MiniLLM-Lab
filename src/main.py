# (entry point)

# Reads input files.

# Sends prompts to the LLM.

# Gets candidate JSON output.

# Calls functions in validation.py to check that output.

# If valid → saves to output/.

# If invalid → logs error and retries.

import json
import sys
from pathlib import Path

from .validation import build_functions_lookup, validate_and_coerce
from .utils import safe_parse_json

def load_json_file(path: Path):
    """Load a JSON file safely, with error handling."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        


        