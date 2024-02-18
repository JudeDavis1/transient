import sys

import torch

from src import logger
from src.config import config
from src.model.transformer import TransientRunner, dataset

WARMUP = True

# torch._C._jit_set_profiling_mode(False)


def main():
    with torch.no_grad():
        device = "mps"
        runner = TransientRunner(
            block_size=config.BLOCK_SIZE,
            n_embd=config.N_EMBD,
            n_layers=config.N_LAYERS,
            n_heads=config.N_HEADS,
        )
        runner.to_device(device)

        fname = sys.argv[2]
        logger.INFO("Loading", fname)

        runner.load(fname, map_location=device)
        runner.model.eval()

        # runner.compile_model()

        # accuracy = runner.score_accuracy(dataset, n_samples=100)
        # logger.info(f"Accuracy: {accuracy}")

        with torch.jit.optimized_execution(True):
            while True:
                context_str = input("> ")
                context = torch.tensor([dataset.encode(dataset.tokenize(context_str))])

                logger.info(context_str, end="")
                runner.generate(
                    context.to(device),
                    max_new_tokens=int(sys.argv[1]),
                    display=True,
                    temperature=0.7,
                    greedy=False,
                )


if __name__ == "__main__":
    main()
