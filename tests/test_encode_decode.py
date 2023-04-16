import pytest

from src.dataset.dataset import BookCorpusDataset


@pytest.mark.parametrize(
    "test_case",
    [
        "hello",
        "hello world",
        "hello world!",
        "hello world!?",
        "some string with 123 numbers",
        "[brackets]",
        "{braces}",
        "\"quotes\"",
        "'quotes'",
        "hello-world",
        "hello-world!",
        "hello-world!?",
        "don't",
    ]
)
def test_encode(test_case, dataset: BookCorpusDataset):
    """Test encoding"""

    tokenized = dataset.tokenize(test_case)
    encoded = dataset.encode(tokenized)
    
    for i in range(len(tokenized)):
        assert dataset.corpus[encoded[i]] == tokenized[i]
