# Scrape bible verses from wol-api
import time
import requests
from typing import Tuple
from tqdm import tqdm


max_books = 66
cache_interval = 10

def getVerse(b: int, c: int, v: int) -> Tuple[dict, int]:
    """Get verse text b: Book, c: Chapter, v: Verse"""
    req = requests.get(f'https://wol-api.onrender.com/api/v1/bibleVerses/getVerse/{b}/{c}/{v}')
    return (req.json(), req.status_code)


verses = []
for book in (t := tqdm(range(1, max_books + 1))):
    n_chapters_req = requests.get(f'https://wol-api.onrender.com/api/v1/bibleVerses/getNumberOfChapters/{book}')
    n_chapters = int(n_chapters_req.json()['data'])

    for chapter in range(1, n_chapters + 1):
        n_verses_req = requests.get(f'https://wol-api.onrender.com/api/v1/bibleVerses/getNumberOfChapters/{book}')
        n_verses = int(n_verses_req.json()['data'])
        
        for verse in range(1, n_verses + 1):
            try:
                t.set_description(f'{book}/{chapter}/{verse}')
                json, status = getVerse(book, chapter, verse)

                verses.append(json['data'])

                # cache verses into a file to save
                if len(verses) >= 
            except:
                continue
