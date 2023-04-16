from src.dataset.dataset import BookCorpusDataset
from tests.dataset.generic_string_test_cases_dataset import generic_string_testcases


@generic_string_testcases()
def test_encode(test_case, dataset: BookCorpusDataset):
    """Test encoding"""

    tokenized = dataset.tokenize(test_case)
    encoded = dataset.encode(tokenized)

    for i in range(len(tokenized)):
        assert dataset.corpus[encoded[i]] == tokenized[i]
