import pytest

from src.config import Config
from src.dataset.dataset import BookCorpusDataset


@pytest.fixture()
def config() -> Config:
    config = Config()
    config.BLOCK_SIZE = 64
    return config


@pytest.fixture()
def dataset(config: Config) -> BookCorpusDataset:
    dataset = BookCorpusDataset(
        folder="data",
        train_data_file="train_data.gz.npy",
        chunk_size=config.BLOCK_SIZE,
    )
    return dataset

@pytest.fixture()
def dataset_with_batches(dataset) -> BookCorpusDataset:
    dataset.generate_batches()
    return dataset


@pytest.fixture()
def dataset_with_batches(dataset) -> BookCorpusDataset:
    dataset.generate_batches()
    return dataset
