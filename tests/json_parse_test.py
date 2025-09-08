from src.utils import safe_parse_json

raw_output = """
Here is what I think:
{
  "prompt": "Add 2 and 3",
  "fn_name": "fn_add_numbers",
  "args": {"a": 2, "b": 3}
}
Thank you!
"""

parsed = safe_parse_json(raw_output)
print(parsed)