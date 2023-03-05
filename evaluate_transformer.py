import sys

import logger
from train_transformer import transformer_model_name
from bigram_transformer import *


with torch.no_grad():
    cpu_device = torch.device('cpu')
    model = BigramLanguageModel()
    model.to_device(cpu_device)
    model.load(transformer_model_name, map_location='cpu')

    # generate from the model
    context = torch.zeros((1, 10), dtype=torch.long, device=cpu_device)

    while True:
        context_str = input('> ')
        context = torch.tensor([dataset.encode(context_str)])

        logger.info(context_str, end='')
        model.generate(context, max_new_tokens=int(sys.argv[1]), display=True)
