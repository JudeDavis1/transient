import sys

from bigram_transformer import *


with torch.no_grad():
    model = BigramLanguageModel()
    model.load_state_dict(torch.load(transformer_model_name))

    # generate from the model
    device = torch.device('cpu')
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.cpu()
    model.generate(context, max_new_tokens=int(sys.argv[1]))
