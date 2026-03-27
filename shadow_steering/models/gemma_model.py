from .base_model import BaseModel


class GemmaModel(BaseModel):
    """
    Gemma-2-2B-IT specific wrapper.

    Extra config fields:
        model.suffix : text appended to user turn (default: ' Answer directly and concisely.')
    """

    def format_prompt(self, prompt: str) -> str:
        suffix = getattr(self.config.model, 'suffix',
                         ' Answer directly and concisely.')
        return (f"<start_of_turn>user\n{prompt}{suffix}"
                f"<end_of_turn>\n<start_of_turn>model\n")
