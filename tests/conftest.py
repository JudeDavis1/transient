import pytest

from src.config import Config
from src.dataset.dataset import BookCorpusDataset
from src.model.transformer import TransientRunner


@pytest.fixture(scope="session")
def small_config() -> Config:
    """Generate a config for a small model"""

    config = Config()
    config.BLOCK_SIZE = 64
    return config


@pytest.fixture(scope="session")
def default_config() -> Config:
    """Generate a config with the default values"""

    return Config()


@pytest.fixture(scope="session", autouse=True)
def dataset() -> BookCorpusDataset:
    """Generate a dataset"""

    dataset = BookCorpusDataset(
        folder="data",
        train_data_file="train_data.gz.npy",
    )
    return dataset


@pytest.fixture(scope="session", autouse=True)
def dataset_with_batches(dataset, small_config: Config) -> BookCorpusDataset:
    """Generate a dataset with batches"""

    dataset.generate_batches(small_config.BLOCK_SIZE)
    return dataset


@pytest.fixture(scope="session")
def model_runner() -> TransientRunner:
    """Return a model runner"""

    model_runner = TransientRunner()
    return model_runner


@pytest.fixture(scope="session", autouse=True)
def pretrained_model(model_runner: TransientRunner) -> TransientRunner:
    """Load a pretrained model"""

    model_runner.load(load_cache='model_cache', map_location='cpu')
    model_runner.model.eval()
    return model_runner
