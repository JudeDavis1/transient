import pytest

from src.dataset.dataset import BookCorpusDataset


@pytest.fixture()
def dataset():
    dataset = BookCorpusDataset(
        folder="data", train_data_file="train_data.gz.npy"
    )
    return dataset