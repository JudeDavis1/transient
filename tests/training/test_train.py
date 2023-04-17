import time
import pytest
import random

from torch.utils.data import DataLoader

from src.config import Config
from src.dataset.dataset import BookCorpusDataset
from src.model.train import get_batch


@pytest.mark.parametrize(
    "batch_size",
    # try different batch sizes
    [random.randint(5, 70) for _ in range(10)]
)
def test_get_batch(
    config: Config,
    dataset_with_batches: BookCorpusDataset,
    batch_size: int
):
    dataloader = DataLoader(
        dataset_with_batches[:100],
        batch_size=batch_size,
        shuffle=True
    )

    start = time.time()
    x, y = get_batch(dataloader)
    end = time.time()

    # check that batch is generated in a reasonable time
    assert end - start < 0.7

    # check shapes are correct
    assert x.shape == (batch_size, config.BLOCK_SIZE)
    assert y.shape == (batch_size, config.BLOCK_SIZE)

