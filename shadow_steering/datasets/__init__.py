from .harmful_prompts_dataset import HarmfulPromptsDataset


def get_data_loader(config):
    data_loaders = {
        'harmful_prompts': HarmfulPromptsDataset,
    }
    return data_loaders[config.dataset.dataset_class](config)
