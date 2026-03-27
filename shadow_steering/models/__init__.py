from .gemma_model import GemmaModel
from .llama_model import LlamaModel


def get_model(config):
    models = {
        'gemma': GemmaModel,
        'llama': LlamaModel,
    }
    return models[config.model.model_class](config)
