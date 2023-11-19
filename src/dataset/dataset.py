"""

The main Dataset class for the entire model is written here.
It keeps track of train_data and the corpus and everything to do
with the book dataset. Also supports multithreading.

"""


import asyncio
import os
import warnings
from typing import Union

import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

from src import logger


class BookCorpusDataset(Dataset):

    """
    Class:
        - The main dataset that contains all the required data for training
        the model.
        - Supports multiprocessing.
    Args:
        folder:
            - The folder to load text files from.
        train_data_file:
            - The filename to load the training data cache from.
    """

    def __init__(
        self,
        folder: str = "data",
        train_data_file: str = "train_data.gz.npy",
    ):
        self.folder = folder
        self.loop = asyncio.get_event_loop()
        self.train_data_file = train_data_file
        self.tokenizer: Tokenizer = Tokenizer.from_file("bpe_model.json")
        self.corpus: list[str] = self.tokenizer.get_vocab().keys()
        self.vocab_size = self.tokenizer.get_vocab_size()

        # the list of data in (train_x, train_y) format
        self.train_data = []
        self.batch_data = []

    def load_dataset(self):
        if os.path.exists(self.train_data_file):
            logger.info(f"Loading training data: {self.train_data_file}")
            self.train_data: np.ndarray = np.load(
                self.train_data_file, allow_pickle=True
            )
            logger.info(self.train_data.shape)
            return

        self.file_contents = self._run_load_corpus(
            folder=self.folder, just_contents=True
        )
        self.train_data = np.array(self.encode(self.file_contents))

        np.save(self.train_data_file, self.train_data)

        logger.info("All elements exist:", all(self.train_data))
        logger.info(len(self.train_data))

    def generate_batches(self, chunk_size):
        if not len(self.train_data):
            raise ValueError("please call load_dataset() before generating batches.")

        self.chunk_size = chunk_size

        for i in range(0, len(self.train_data) - self.chunk_size, self.chunk_size):
            sample = self.get_batch(i)

            # drop any remaining data that doesn't fit chunk_size
            if len(sample[0]) != self.chunk_size or len(sample[1]) != self.chunk_size:
                break

            # add pairs
            self.batch_data.append(sample)

    def get_batch(self, idx):
        starting_phrase = self.train_data[idx : idx + self.chunk_size]
        target_word = self.train_data[idx + 1 : idx + 1 + self.chunk_size]

        return (starting_phrase, target_word)

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def encode(self, s):
        if isinstance(s, list):
            return [t.ids for t in self.tokenizer.encode_batch(s)][0]
        return self.tokenizer.encode(s).ids

    def decode(self, l):
        return self.tokenizer.decode(l)

    def _run_load_corpus(self, folder="data", just_contents=False):
        return self.loop.run_until_complete(
            load_corpus(text_file_dir=folder, just_contents=just_contents)
        )

    def __getitem__(self, index):
        return self.batch_data[index]

    def __len__(self):
        return len(self.batch_data)


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
