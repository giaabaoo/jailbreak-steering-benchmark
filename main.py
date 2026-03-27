from shadow_steering.pipelines import get_pipeline
from shadow_steering.utils import launch, setup_config


def main(config):
    pipeline = get_pipeline(config)
    pipeline.run()


if __name__ == '__main__':
    config = setup_config()
    main(config)
