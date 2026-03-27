from .jailbreak_evaluator import JailbreakEvaluator


def get_evaluator(config):
    evaluators = {
        'jailbreak': JailbreakEvaluator,
    }
    return evaluators[config.evaluator.evaluator_class](config)
