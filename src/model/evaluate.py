import sys

from src import logger
from src.config import Config
from src.model.transformer import *


WARMUP = True

def main():
    with torch.no_grad():
        device = "mps"
        runner = TransientRunner(
            block_size=Config.BLOCK_SIZE,
            n_embd=Config.N_EMBD,
            n_layers=Config.N_LAYERS,
            n_heads=Config.N_HEADS,
        )
        runner.to_device(device)

        fname = sys.argv[2]
        logger.INFO("Loading", fname)
        
        runner.load(fname, map_location=device)
        runner.model.eval()

        runner.compile_model()

        # accuracy = runner.score_accuracy(dataset, n_samples=100)
        # logger.info(f"Accuracy: {accuracy}")

        while True:
            context_str = input("> ")
            context = torch.tensor([dataset.encode(dataset.tokenize(context_str))])

            logger.info(context_str, end='')
            runner.generate(
                context.to(device),
                max_new_tokens=int(sys.argv[1]),
                display=True,
                temperature=0.2,
                greedy=False,
            )


if __name__ == "__main__":
    main()
