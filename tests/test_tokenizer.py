"""Collection of tokenizer tests"""

import pytest

from src.dataset.dataset import BookCorpusDataset


class TestTokenizer:
    @pytest.mark.parametrize(
        "test_case,expected",
        [
            ("hello", ["hello"]),
            ("hello world", ["hello", " ", "world"]),
            ("hello world!", ["hello", " ", "world", "!"]),
            ("hello world!?", ["hello", " ", "world", "!?"]),
            ("some string with 123 numbers", ["some", " ", "string", " ", "with", " ", "123", " ", "numbers"]),
            ("(parenthesis)", ["(", "parenthesis", ")"]),
            ("[brackets]", ["[", "brackets", "]"]),
            ("{braces}", ["{", "braces", "}"]),
            ("\"quotes\"", ["\"", "quotes", "\""]),
            ("'quotes'", ["'", "quotes", "'"]),
            ("hello-world", ["hello", "-", "world"]),
            ("hello-world!", ["hello", "-", "world", "!"]),
            ("hello-world!?", ["hello", "-", "world", "!?"]),
            ("don't", ["don", "'", "t"]),
        ],
    )
    def test_single_phrase(self, dataset: BookCorpusDataset, test_case, expected):
        """Test tokenizer on single phrases"""

        self._generic_tokenizer_test(dataset, test_case, expected)
    
    
    @pytest.mark.parametrize(
        "test_case,expected",
        [
            ("hello world. hello world.", ["hello", " ", "world", ".", " ", "hello", " ", "world", "."]),
            ("hello world! hello world!", ["hello", " ", "world", "!", " ", "hello", " ", "world", "!"]),
            ("hello world? hello world?", ["hello", " ", "world", "?", " ", "hello", " ", "world", "?"]),
            ("hello world. hello world!", ["hello", " ", "world", ".", " ", "hello", " ", "world", "!"]),
            ("hello world. hello world?", ["hello", " ", "world", ".", " ", "hello", " ", "world", "?"]),
            ("hello world! hello world?", ["hello", " ", "world", "!", " ", "hello", " ", "world", "?"]),
            ("hello world. hello world! hello world?", ["hello", " ", "world", ".", " ", "hello", " ", "world", "!", " ", "hello", " ", "world", "?"]),
        ]
    )
    def test_multi_phrase(self, dataset: BookCorpusDataset, test_case, expected):
        """Test tokenizer on multiple sentences"""

        self._generic_tokenizer_test(dataset, test_case, expected)

    def _generic_tokenizer_test(self, dataset: BookCorpusDataset, test_case, expected):
        assert dataset.tokenize(test_case) == expected
    
