
from src.dataset.dataset import BookCorpusDataset


@test_cases()
def test_encode(test_case, dataset: BookCorpusDataset):
    """Test encoding"""

    tokenized = dataset.tokenize(test_case)
    encoded = dataset.encode(tokenized)

    for i in range(len(tokenized)):
        assert dataset.corpus[encoded[i]] == tokenized[i]
