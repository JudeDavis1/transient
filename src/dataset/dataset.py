"""

The main Dataset class for the entire model is written here.
It keeps track of train_data and the corpus and everything to do
with the book dataset. Also supports multithreading.

"""


import asyncio
import os
import string
import warnings
from tokenizers import Tokenizer
from typing import Optional, Union

import nltk
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

from src import logger


class BookCorpusDataset(Dataset):

    """
    Class:
        - The main dataset that contains all the required data for training
        the model.
        - Supports multiprocessing.
    Args:
        chunk_size:
            - The amount of words in a batch.
            - This is set to None, when just_corpus=True.
        just_corpus:
            - Whether the dataset should only prepare the corpus (when not training).
            - You can't run generate_batches() if this is set to True.
        save_corpus:
            - Whether to save the corpus in a file or not.
        cache_train_data:
            - Whether or not to save the training data instead of processing it every time
            at runtime.
        train_data_file:
            - The filename to load the training data from.
        corpus_from_file:
            - The filename to load the corpus from.
    """

    def __init__(
        self,
        folder="data",
        train_data_file: Optional[str] = None,
        just_corpus=False,
    ):
        nltk.download("punkt", quiet=True)

        self.loop = asyncio.get_event_loop()
        self.train_data_file = (
            train_data_file if train_data_file else "train_data.gz.npy"
        )
        # self.tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+|\s+|[^\w\s]+")
        self.tokenizer = Tokenizer.from_file("bpe_model.json")
        self.file_contents = self._run_load_corpus(folder=folder, just_contents=True)
        tokenized = self.tokenize(self.file_contents)
        self.corpus = self.tokenizer.get_vocab().keys()
        self.vocab_size = self.tokenizer.get_vocab_size()

        # sets of features and labels
        self.x_data = []
        self.y_data = []

        if just_corpus:
            return

        # the list of data in (train_x, train_y) format
        self.prep_data = []
        if os.path.exists(self.train_data_file):
            logger.info(f"Loading training data: {self.train_data_file}")
            self.train_data: np.ndarray = np.load(self.train_data_file, allow_pickle=True)
            logger.info(self.train_data.shape)
            return

        self.limit = float("inf")
        self.train_data = np.array(self.encode(self.file_contents, self.limit))

        np.save(self.train_data_file, self.train_data)
        logger.info("All elements exist:", all(self.train_data))
        logger.info(len(self.train_data))

    def generate_batches(self, chunk_size):
        self.chunk_size = chunk_size

        beginning = 0
        next_idx = self.chunk_size

        while True:
            sample = self.get_batch(beginning, next_idx)
            if len(sample[0]) != self.chunk_size or len(sample[1]) != self.chunk_size:
                break

            # add features
            self.x_data.append(sample[0])

            # add labels
            self.y_data.append(sample[1])

            # add pairs
            self.prep_data.append(sample)

            beginning = next_idx
            next_idx = beginning + self.chunk_size

    def get_batch(self, beginning, next_idx):
        starting_phrase = self.train_data[beginning:next_idx]
        target_word = self.train_data[next_idx : next_idx + self.chunk_size]

        return (starting_phrase, target_word)

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def encode(self, s, limit=float("inf")):
        # l_idx = []
        # i = 0
        # s = s if limit == float("inf") else s[:limit]
        # for token in tqdm(s):
        #     try:
        #         token = self.corpus.index(token)
        #     except ValueError:
        #         token = self.corpus.index(token.lower())

        #     l_idx.append(token)
        #     i += 1

        #     if i >= limit:
        #         break

        # return l_idx
        if isinstance(s, list):
            return [t.ids for t in self.tokenizer.encode_batch(s)][0]
        return self.tokenizer.encode(s).ids

    def decode(self, l, idx=True):
        # if idx:
        #     return self.corpus[l]

        # return [self.corpus[idx] for idx in l]
        return self.tokenizer.decode(l)

    def _run_load_corpus(self, folder="data", just_contents=False):
        return self.loop.run_until_complete(
            load_corpus(text_file_dir=folder, just_contents=just_contents)
        )

    def __getitem__(self, index):
        return self.prep_data[index]

    def __len__(self):
        return len(self.prep_data)


async def load_corpus(text_file_dir, **kwargs) -> Union[list, str]:
    corpus = ""
    files_str = os.listdir(text_file_dir)
    files = [
        open(os.path.join(text_file_dir, f), "r", encoding="utf-8") for f in files_str
    ]

    logger.INFO("Collecting tokens from:\n")
    files_str.sort(key=len)

    for c in files_str:
        logger.info(c)
    print()

    for f in files:
        corpus += f.read()

    return corpus
