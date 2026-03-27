from .inference_only import InferenceOnly
from .baseline import Baseline
from .evaluate import Evaluate


def get_pipeline(config):
    pipelines = {
        'inference_only': InferenceOnly,
        'baseline':       Baseline,
        'evaluate':       Evaluate,
    }
    return pipelines[config.pipeline.name](config)
