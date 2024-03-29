"""Scrape bible verses from wol-api"""

from typing import Tuple

import requests
from tqdm import tqdm

from src import logger

first_book = 50
max_books = 66


def main():
    pbar = tqdm(range(first_book, max_books + 1))
    for book in pbar:
        n_chapters_req = requests.get(
            f"https://wol-api.onrender.com/api/v1/bibleVerses/getNumberOfChapters/{book}"
        )
        n_chapters = int(n_chapters_req.json()["data"])

        for chapter in range(1, n_chapters + 1):
            verses_json = requests.get(
                f"https://wol-api.onrender.com/api/v1/bibleVerses/getVersesInChapter/{book}/{chapter}"
            )
            verses = verses_json.json()["data"]

            try:
                pbar.set_description(f"{book}/{chapter}")

                # cache verses into a file to save
                save_verses("\n".join(verses))
            except KeyboardInterrupt:
                logger.info("Exiting")
                exit(0)
            except:
                continue


def getVerse(b: int, c: int, v: int) -> Tuple[dict, int]:
    """Get verse text: b: Book, c: Chapter, v: Verse"""
    req = requests.get(
        f"https://wol-api.onrender.com/api/v1/bibleVerses/getVerse/{b}/{c}/{v}"
    )
    return (req.json(), req.status_code)


def save_verses(verses: list):
    with open("./bible_verses.txt", "a") as f:
        f.writelines(verses)


if __name__ == "__main__":
    main()
