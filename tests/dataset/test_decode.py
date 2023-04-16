from src.dataset.dataset import BookCorpusDataset
from tests.dataset.generic_string_test_cases_dataset import generic_string_testcases


@generic_string_testcases()
def test_decode(test_case, dataset: BookCorpusDataset):
    """Test decoding"""

    tokenized = dataset.tokenize(test_case)
    encoded = dataset.encode(tokenized)
    decoded = dataset.decode(encoded, idx=False)

    assert decoded == tokenized
