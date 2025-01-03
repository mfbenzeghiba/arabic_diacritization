"""Process the data for teshkeel.
    some methods are from:
    https://github.com/AliOsm/arabic-text-diacritization/blob/master/helpers/pre_process_tashkeela_corpus.ipynb
"""

import os
import random
import  argparse
from collections import defaultdict
from typing import List, Dict
from multiprocessing import Pool

import pandas as pd
import regex

from pyarabic import araby

import clean_funcs as cf
import utils as ut


def parse_args (args: list = None) -> argparse.Namespace:
    """Parse the arguments."""

    parser = argparse.ArgumentParser(description='Extract diacritized arabic text from html pages.')
    parser.add_argument('--file_list', type=str,
            help='File containing the train list with extension lst or a single file')
    parser.add_argument('--keep_digit', action='store_true',
            help='Keep the digits in the cleaned text, and convert them to constant number.')
    parser.add_argument('--keep_punctuation', action='store_true',
            help='Keep the punction in the cleaned text and convert them to a single punctuation.')
    parser.add_argument('--dtoc_ratio', type=float, default=0.6,
            help='Keep only sentences with diacritics to char ratio higher than this value.')
    parser.add_argument('--out_dir', type=str,
            help='Folder to save the processed text.')
    parser.add_argument('--out_file', type=str,
            help='output file that contains the processed text.')
    args = parser.parse_args()
    return args


def create_folder(path: str) -> None:
    """Create a directory if it does not exist.

    Args:
        path (str): The directory path
    """

    if not os.path.isdir(path):
        os.makedirs(path)

def process_line(text_line: str, opts: argparse.Namespace) -> str:
    """Process a single line of text.

    Args:
        text (str): a line of text

    Returns:
        str: the processed line
    """

    # remove white space at the begining and end.
    text_line = text_line.strip()
    # split paragraphs into sentences.
    sentences = araby.sentence_tokenize(text_line)

    processed_line = []

    for _, s in enumerate(sentences):
        # Do not process sentences that do not contains arabic text
        if not regex.search(ut.ARABIC_LETTER_PATTERN, s):
            continue

        # remove non arabic characters
        s = regex.sub('[\p{Latin}|({Latin})]', '', s)

        # Normalize arabic characters to theirs siolated forms
        s = araby.normalize_ligature(s)

        # remove lines with non valid characters
        s = cf.remove_non_valid_char(s)

        # remove punctuation
        if not opts.keep_punctuation:
            s = cf.remove_punctuation(s)
        else:
            s = cf.normalize_punctuation(s)

        # remove digits
        if not opts.keep_digit:
            s = cf.remove_digits(s)
        else:
            s = cf.normalize_digits(s)

        # remove sentences that containes few diacritized words
        if not cf.diacritic_to_char_ratio(s, opts.dtoc_ratio):
            continue

        # remove extra spaces
        s = ' '.join(s.split())

        processed_line.append(s)

    return processed_line

def merge_files(file_list: List[str]) -> Dict[int, List[str]]:
    """Merge sentences with the same length.

    Args:
        text_list (list): _description_
    """

    cl_text = defaultdict(list)
    for text_list in file_list:
        for s in text_list:
            cl_text[len(s.split())].append(s)
    for key in cl_text:
        cl_text[key] = list(set(cl_text[key]))
    return cl_text

def process_file(raw_file: str, opts: argparse.Namespace) -> List[str]:
    """Clean the raw text file.

    Args:
        raw_file (str): The raw text file
    """

    processed_file = []
    print(f'Process {raw_file}...')
    with open(raw_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for _, text_line in enumerate(lines):
        processed_line = process_line(text_line, opts)
        # add the processed sentence
        processed_file.extend(processed_line)
    return processed_file

def main(args: argparse.Namespace):
    """Process the data

    Args:
        args (dict): The options
    """

    raw_text = args.file_list
    cleaned_files = []

    if os.path.isdir(raw_text):
        # search for all text files
        raw_files = [os.path.join(raw_text,f) for f in os.listdir(raw_text) if f.endswith('.txt') ]
        print('Cleaning The data...')
        pool = Pool(processes=10)
        processes = [
            pool.apply_async(process_file, args=(raw_file, args)) for raw_file in raw_files
        ]
        cleaned_files = [p.get() for p in processes]
        pool.close()
    elif os.path.isfile(raw_text):
        cleaned_files = [process_file(raw_text, args)]

    print('Merge the data...')
    cleaned_text = merge_files(cleaned_files)

    print('Split long sentences...')
    # split long sentences into smaller chunks
    # and create the final text with diacritics
    cleaned_text = cf.splited_sentences(cleaned_text, args.dtoc_ratio)

    # twd: text with diacritics
    twd = list(set([s for k in cleaned_text for s in cleaned_text[k]]))
    data = {
        'text_w_diac': twd
    }
    fdata = pd.DataFrame(data)

    create_folder(args.out_dir)
    fname = os.path.join(args.out_dir, args.out_file)
    fdata.to_csv(fname, index=False)

    print(f'The file: {fname} is created...')
    print(f'{len(fdata)} sentences have created..')

if __name__ == '__main__':

    random.seed(10)
    options = parse_args()
    main(options)
    print('Processing finished...')
