"""Collection of dataset tests"""

import pytest

from ..src.dataset.dataset import BookCorpusDataset


@pytest.fixture()
def dataset():
    dataset = BookCorpusDataset(
        folder="../data", train_data_file="../train_data.gz.npy"
    )
    return dataset


class TestTokenizer:
    @pytest.mark.parametrize(
        "test_case,expected",
        [
            ("hello", ["hello"]),
            ("hello world", ["hello", " ", "world"]),
        ],
    )
    def test_single_phrase(self, dataset: BookCorpusDataset, test_case, expected):
        assert dataset.tokenize(test_case) == expected
