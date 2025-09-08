import json
from typing import Any, Optional

def extract_json_substring(text: str) -> Optional[str]:
    """
    Extract the first balanced {...} substring from text.
    Returns None if no valid JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def safe_parse_json(text: str) -> Optional[Any]:
    """
    Extract a JSON substring and parse it into a Python object.
    Returns None if extraction or parsing fails.
    """
    snippet = extract_json_substring(text)
    if snippet is None:
        return None
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None
            