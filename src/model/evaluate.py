import sys

from .. import logger
from ..config import Config
from .transformer import *


def main():
    with torch.no_grad():
        cpu_device = torch.device("cpu")
        model = TransformerModel(
            block_size=Config.BLOCK_SIZE,
            n_embd=Config.N_EMBD,
            n_heads=Config.N_HEADS,
            n_layers=Config.N_LAYERS,
        )
        model.to_device(cpu_device)
        model.load(map_location=cpu_device, load_cache=True)
        model.eval()

        while True:
            context_str = input("> ")
            context = torch.tensor([dataset.encode(dataset.tokenize(context_str))])

            logger.info(context_str)
            model.generate(context, max_new_tokens=int(sys.argv[1]), display=True)


if __name__ == "__main__":
    main()
