import sys

from src import logger
from src.config import Config
from src.model.transformer import *


def main():
    with torch.no_grad():
        cpu_device = torch.device("cpu")
        runner = TransientRunner(
            block_size=Config.BLOCK_SIZE,
            n_embd=Config.N_EMBD,
            n_layers=Config.N_LAYERS,
            n_heads=Config.N_HEADS,
        )
        runner.to_device(cpu_device)
        runner.model.eval()
        runner.load("model_cache", map_location=cpu_device)

        while True:
            context_str = input("> ")
            context = torch.tensor([dataset.encode(dataset.tokenize(context_str))])

            logger.info(context_str)
            runner.generate(context, max_new_tokens=int(sys.argv[1]), display=True)


if __name__ == "__main__":
    main()
