import numpy as np
from typing import List
from llm_sdk import Small_LLM_Model


def generate_text(
    prompt: str,
    model: Small_LLM_Model,
    tokenizer,
    max_tokens: int = 100,
    stop_token: str = "}"
) -> str:
    """
    Generate text from a prompt using autoregressive decoding.
    For now, uses greedy decoding (argmax).
    """
    # Encode initial prompt
    input_ids: List[int] = tokenizer.encode(prompt)
    
    for i in range(max_tokens):
        # Get logits for next token
        logits = model.get_logits_from_input_ids(input_ids)
        
        # Pick most likely token (argmax)
        next_id = int(np.argmax(logits))
        input_ids.append(next_id)
        
        # Decode current sequence
        text = tokenizer.decode(input_ids)
        text = text.replace("Ġ", " ").replace("Ċ", "\n")
        
        if stop_token in text:
            break

    return text