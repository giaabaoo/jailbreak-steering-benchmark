from .base_model import BaseModel


class LlamaModel(BaseModel):
    """
    Llama-3.1-8B-Instruct specific wrapper.

    Uses apply_chat_template for formatting.
    Standard KV-cache — no HybridCache issues.
    """

    def format_prompt(self, prompt: str) -> str:
        msgs = [{'role': 'user', 'content': prompt}]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
