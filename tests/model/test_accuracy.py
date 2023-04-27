from torch.utils.data import DataLoader

from src.dataset.dataset import BookCorpusDataset
from src.model.transformer import TransientRunner

def test_accuracy(pretrained_model: TransientRunner, dataset: BookCorpusDataset):
    """Test a pretrained model's accuracy"""

    acc = pretrained_model.score_accuracy(dataset)
    print(acc)
