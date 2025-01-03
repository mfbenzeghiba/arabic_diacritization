"""Methods and functions to clean data."""

import sys
from pathlib import Path

from collections import defaultdict
from typing import List, Dict

import regex
import numpy as np
from bs4 import BeautifulSoup

from diacritization_evaluation import util as de

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import data_processing.utils as dut


def fix_diacritics(text: str) -> str:
    """Fix some issues with diacritics.

    Args:
        text (str): text

    Returns:
        str: text
    """

    # Replace consecutive diacritics with a single diacritic
    text = regex.sub(r'([\u064b-\u0652])\1+', r'\1', text)

    # Ending Diacritics: Remove diacritics at the end of a word
    text = regex.sub(r'(\S)[\u064b-\u0652](\s+)', r'\1\2', text)

    # Misplaced Diacritics: Remove spaces between characters and diacritics
    text = regex.sub(r'(\S)\s+([\u064b-\u0652])', r'\1\2', text)

    return text


def remove_non_valid_char(text: str) -> str:
    """Remove other non valid characters.

    Args: 
        text (str): text

    Returns:
        str: text
    """
    text = text.replace('ﺈ', '\u0625' )
    text = regex.sub(dut.NON_VALID_CHARACTERS_PATTERN, '', text)
    return text


def remove_punctuation(text: str) -> str:
    """Remove punctuation.

    Args:
        text (str): text

    Returns:
        str: text
    """

    text = regex.sub(dut.PUNCTUATIONS_PATTERN, '', text)
    return text


def normalize_punctuation(text: str) -> str:
    """Replace punctuation marks with the letter p.

    Args:
        text (str): text

    Returns:
        str: text
    """

    text = regex.sub(dut.PUNCTUATIONS_PATTERN, 'p', text)
    return text


def remove_arabic_letters(text: str) -> str:
    """Remove arabic letters

    Args:
        text (str): _description_:wq

    """

    text = regex.sub(dut.ARABIC_LETTER_PATTERN, '', text)
    return text


def remove_digits(text: str) -> str:
    """Remove digits

    Args:
        text (str): text

    Returns:
        str: text without digits
    """

    text = regex.sub(dut.DIGIT_PATTERN, '', text)
    return text


def normalize_digits(text: str) -> str:
    """Replace digits by the letter d

    Args:
        text (str): text

    Returns:
        str: text where digit sequences are replaced by d
    """

    text = regex.sub(dut.DIGIT_PATTERN, 'd', text)
    return text


def remove_diacritics(text: str) -> str:
    """Remove diacritics.

    Args:
        text (str): text with diacritics

    Returns:
        str: text without diacritics
    """

    text = regex.sub(dut.DIACRITIC_PATTERN, '', text)
    return text


def remove_html_tags(text: str) -> str:
    """Remove HTML tags.

    Args:
        content (str): a text

    Returns:
        str: The processed text without html tags
    """

    soup = BeautifulSoup(text, 'html_parser')
    return soup.get_text()


def diacritic_to_char_ratio(text: str, thre: float=0.6) -> bool:
    """Compute the ratio between diacritics and characters.

    Args:
        text (str): The text
        threshold (float): The threshold to accep/reject the text

    Returns:
        bool: accept or not the sentence 
    """

    # remove space:
    text = ''.join(text.split())
    try:
        _, twod, diacs = de.extract_haraqat(text)
        diacs = [d if d else '~' for d in diacs]
        for i, c in enumerate(twod):
            # do not count the long vowels
            if c in ['و', 'ي', 'ا' ] and diacs[i] == '~':
                twod[i] = '~'
        twod = [d for d in twod if d != '~']
        diacs = [d for d in diacs if d != '~']

        if len(diacs)/len(twod) < thre:
            return False
        else:
            return True
    except:
        return False


def remove_duplicate(text: list) -> list:
    """Remove duplicate from a list

    Args:
        text (list): list of words

    Returns:
        list: list of words without duplicates
    """

    return list(set(text))


def splited_sentences(text_dict: Dict[int, List[str]], thre:float) -> Dict[int, List[str]]:
    """Split sentences with a char to diacrics ratio higher than threshold into chunks.
       Only sentence with length heigher than 15 words will be split.

    Args:
        text_dict (dict): Dictionary with list of sentences, the keys are the
                          length of sentences in the list.
        thre (float): The threshold to split a sentence.
    
    Returns:
        a dictionary with list of sentences.
    """

    sd = defaultdict(list)
    for  k in text_dict:
        if k <= 15:
            for s2 in text_dict[k]:
                if not diacritic_to_char_ratio(s2, thre):
                    continue
                sd[k].append(s2)
        else:
            for s1 in text_dict[k]:
                s1 = s1.split()
                seq_len = np.random.randint(15, 60, 1)[0]
                ss = [' '.join(s1[i:i+seq_len]) for i in range(0, len(s1), seq_len)]
                for s2 in ss:
                    if not diacritic_to_char_ratio(s2, thre):
                        continue
                    sd[len(s2.split())].append(s2)
    for key in sd:
        sd[key] = list(set(sd[key]))
    return sd
