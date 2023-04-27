import pytest


def generic_string_testcases(param="test_case"):
    """Generate test cases for encoding and decoding"""

    return pytest.mark.parametrize(
        param,
        [
            "hello",
            "hello world",
            "hello world!",
            "hello world!?",
            "some string with 123 numbers",
            "[brackets]",
            "{braces}",
            '"quotes"',
            "'quotes'",
            "hello-world",
            "hello-world!",
            "hello-world!?",
            "don't",
            "don't do it",
            "don't do it!",
            "don't do it!?",
        ],
    )
