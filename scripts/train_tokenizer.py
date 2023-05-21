"""
Train the BPE tokenizer on the corpus.

Note:
    This script needs to be run from the root directory of the project.
"""


from tokenizers import Tokenizer, pre_tokenizers, ByteLevelBPETokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


tokenizer = ByteLevelBPETokenizer()

tokenizer.train(vocab_size=50_000, files=["data/wiki.txt"], min_frequency=2)

print("[*] Saving the tokenizer...")
tokenizer.save("bpe_model.json")

print("[*] Checking if the tokenizer can be loaded...")
tokenizer = Tokenizer.from_file("bpe_model.json")
print("[*] Tokenizer loaded successfully!")
