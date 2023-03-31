import sys

import config
import logger

from .transformer import *


with torch.no_grad():
    cpu_device = torch.device("cpu")
    model = TransformerModel(
        block_size=config.BLOCK_SIZE,
        n_embd=config.N_EMBD,
        n_head=config.N_HEAD,
        n_layers=config.N_LAYERS,
    )
    model.to_device(cpu_device)
    model.load(map_location=cpu_device, load_cache=True)
    model.eval()

    while True:
        context_str = input("> ")
        context = torch.tensor([dataset.encode(dataset.tokenize(context_str))])

        logger.info(context_str)
        model.generate(context, max_new_tokens=int(sys.argv[1]), display=True)
