import numpy as np
import torch
import os
from transformers import PreTrainedModel, PreTrainedTokenizer


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(
    model: PreTrainedModel, 
    tokenizer, 
    checkpoint_path, 
    epoch='best'
):
    torch.save(model.state_dict(), os.path.join(checkpoint_path, f"model_{epoch}.pt"))
    # save configuration
    model.config.to_json_file(os.path.join(checkpoint_path, 'config.json'))
    # save exact vocabulary utilized
    tokenizer.save_vocab(checkpoint_path)