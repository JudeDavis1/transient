from src.dataset.dataset import BookCorpusDataset

from .generic_string_test_cases import test_cases


@test_cases()
def test_decode(test_case, dataset: BookCorpusDataset):
    """Test decoding"""

    tokenized = dataset.tokenize(test_case)
    encoded = dataset.encode(tokenized)
    decoded = dataset.decode(encoded, idx=False)

    assert decoded == tokenized
