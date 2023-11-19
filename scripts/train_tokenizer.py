"""
Train the BPE tokenizer on the corpus.

Note:
    This script needs to be run from the root directory of the project.
"""

import glob

from tokenizers import ByteLevelBPETokenizer, Tokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(vocab_size=4096, files=glob.glob("data/*"), min_frequency=2)

print("[*] Saving the tokenizer...")
tokenizer.save("bpe_model.json")

print("[*] Checking if the tokenizer can be loaded...")
tokenizer = Tokenizer.from_file("bpe_model.json")
print("[*] Tokenizer loaded successfully!")
