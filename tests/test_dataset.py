"""Collection of dataset tests"""

import pytest

from ..src.dataset import BookCorpusDataset


@pytest.fixture()
def dataset():
    dataset = BookCorpusDataset()
    return dataset


class TestTokenizer:
    def single_word(self, dataset: BookCorpusDataset):
        assert True
