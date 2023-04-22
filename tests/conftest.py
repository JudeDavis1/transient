import pytest

from src.config import Config
from src.dataset.dataset import BookCorpusDataset


@pytest.fixture(scope="session")
def config() -> Config:
    config = Config()
    config.BLOCK_SIZE = 64
    return config


@pytest.fixture(scope="session")
def dataset(config: Config) -> BookCorpusDataset:
    dataset = BookCorpusDataset(
        folder="data",
        train_data_file="train_data.gz.npy",
        chunk_size=config.BLOCK_SIZE,
    )
    return dataset

@pytest.fixture(scope="session")
def dataset_with_batches(dataset) -> BookCorpusDataset:
    dataset.generate_batches()
    return dataset

