import re

def remove_think_tags(text):
    """Remove <think>...</think> tags from LLM responses."""
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text
