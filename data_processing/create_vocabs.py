"""Create the vocabular."""

import os
from typing import Dict
import argparse
import unicodedata


def parse_args(args: list = None) -> argparse.Namespace:
    """Parse the argument."""

    parser = argparse.ArgumentParser(description='Extract diacritized arabic text from html pages.')
    parser.add_argument('--file_list', type=str, help='The processed csv file.')

    args = parser.parse_args()
    return args

def process_line(text_line: str):
    """Process a text line.

    Args:
        text_line (str): a line of text
    """

    freq = {}
    for c in text_line:
        freq[c] = freq.get(c, 0) + 1
    return freq


def process_file(raw_file: str) -> Dict[str, int]:
    """Clean the raw text file.

    Args:
        raw_file (str): Path to the raw text file
    """

    c_freq = {}
    print(f'Process {raw_file}...')
    with open(raw_file, 'r', encoding='utf-8') as f:
        for text_line in f.readlines():
            c_freq.update(process_line(text_line))
    return c_freq


def main(args: argparse.Namespace):
    """Create the vocabulary.

    Args:
        args (dict): options
    """

    # read the csv file
    raw_files = []
    char_frequencies = {}

    raw_text = args.file_list
    if os.path.isdir(raw_text):
        # search for all text files
        raw_files = [os.path.join(raw_text,f) for f in os.listdir(raw_text) if f.endswith('.txt')]
        for raw_file in raw_files:
            char_frequencies.update(process_file(raw_file))
    elif os.path.isfile(raw_text):
        char_frequencies = process_file(raw_text)

    chars  = list(sorted(char_frequencies.keys()))
    for char in chars:
        print(
            #f'{char}\t{"%04x" %ord(char)}\t{unicodedata.category(char)}\t{unicodedata.name(char)}'
            f'{char}\t{ord(char)}\t{unicodedata.category(char)}\t{unicodedata.name(char)}'
        )


if __name__ == '__main__':

    options = parse_args()
    main(options)
