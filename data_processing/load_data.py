"""data provider for teshkeel data."""

import sys
from pathlib import Path
import random
from typing import List, Tuple

import regex
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from diacritization_evaluation import util

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from data_processing.clean_funcs import remove_digits

class Tokenizer():
    """Create timit vocabulary."""
    def __init__(self, input_vocab: str):
        """Create tashkeel tokenizer

        Args:
            input_vocab (str): Path to the vocab file
        """

        self.vocab_file = input_vocab
        self.char2index = {}
        self.index2char = {}
        self.build_vocab()

    def build_vocab(self) -> None:
        """Read the vocab file and create the character tokenizer."""

        print(f'--> Reading chararacter vocabulary file: {self.vocab_file}')
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                fields = line.strip().split('\t')
                char = fields[0]
                self.index2char[idx] = char
                self.char2index[char] = idx
        print(f'--> The size of the character (input) vocabulary: {len(self)}')

    def __len__(self) -> int:
        """Return the length of the vocabulary.

        Returns:
            int: Number of tokens
        """

        return len(self.index2char)


    def encode(self, sentence: str) -> torch.Tensor:
        """Convert a sequence of token to a sequence of token index

        Args:
            sentence (str): The text to be encoded.

        Returns:
            torch.tensor: 
        """

        sentence = regex.sub(r'\s+', '~', sentence)
        vector = [self.char2index[char] for char in sentence]
        return torch.LongTensor(vector)


    def decode(self, labs: torch.Tensor) -> List[str]:
        """Convert a sequence of token index to a text.

        Args:
            labs (torch.tensoer): a tensor of token index.

        Returns:
            List: sequence of tokens
        """

        chars = [self.index2char[idx] for _, idx in enumerate(labs)]
        return chars


class Diacritizer():
    """Create timit vocabulary."""

    def __init__(self, diac_path: str, no_diac: str = '~'):
        """Create teshkeel diacritizer

        Args:
            diac_vocab (str): Path to the diacritics file.
            no_diac (str): symbole to represent the no diacritic.
        """

        self.diac_vocab = diac_path
        self.no_diac = no_diac
        self.diac2index = {}
        self.index2diac = {}
        self.build_diac_vocab()


    def build_diac_vocab(self) -> None:
        """Build the diacritic vocabulary."""

        print(f'--> Reading diacritics vocabulary file: {self.diac_vocab}')
        with open(self.diac_vocab, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f.readlines()):
                diac = line.strip().split('\t')[0]
                self.diac2index[diac] = index
                self.index2diac[index] = diac
        print(f'--> The size of the diacritics (output) vocabulary: {len(self)}', flush=True)

    def __len__(self) -> int:
        """Return the length of the vocab

        Returns:
            int: Number of tokens
        """
        return len(self.index2diac)


    def encode(self, diacs: str) -> torch.Tensor:
        """Convert a sequence of diacritics to a sequence of their indices

        Args:
            diacs (str): sequence of diacritics to be encoded
        """

        diacs = [d if d else self.no_diac for d in diacs]
        diac_vector = [self.diac2index[diac] for diac in diacs]
        return torch.LongTensor(diac_vector)


    def decode(self, labs: List) -> List[str]:
        """Convert a sequence of indices to a sequence of diacritics.

        Args:
            labs (torch.tensoer): Tensor of diacritic indixes.

        Returns:
            List: a text, sequence of tokens
        """

        diacs = [self.index2diac[idx] for _, idx in enumerate(labs)]
        return diacs


class Tashkeel(Dataset):
    """Create the main Tashkeel class."""

    def __init__(self, data_file: str, tokenizer: Tokenizer,
                 diacritizer: Diacritizer, partial_prob: float=0.0) -> None:

        with open(data_file, 'r', encoding='utf-8') as f:
            self._data_samples = f.readlines()[1:]

        self._tokenizer = tokenizer
        self._diacritizer = diacritizer
        assert partial_prob >= 0 and  partial_prob <= 1.,\
             "Probability of applying partial diacritization should be between 0 and 1."
        self.partial_prob = partial_prob
        self.remove_digits = True


    def __len__(self) -> int:
        """Return the length of the data

        Returns:
            int: Number of sentences
        """

        return len(self._data_samples)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the sample (text without diacritics) and targets (sequence of diacritics)

        Args:
            index (int): The index of the sample

        Returns:
            tuple: The wavfile and the corresponding phone transcription
        """

        # twd: text with diacritics
        # twod: text without diacritics
        twd = self._data_samples[index]
        if self.remove_digits:
            twd = remove_digits(twd)
            twd = ' '.join(twd.split())
        _, twod, diacs = util.extract_haraqat(twd)
        if len(twod) != len(diacs):
            print('There is an issue with the text')

        in_diacs = [''] * len(diacs)
        apply_partial = np.random.choice([True, False], p=[self.partial_prob, 1-self.partial_prob])
        if apply_partial:
            mask_prob = round(np.random.uniform(0, 1), 1)
            nb_masked_diacs = int(mask_prob * len(diacs))
            masked_diacs_idx = np.random.choice(range(len(diacs)), nb_masked_diacs, replace=False)
            in_diacs = ['' if i in masked_diacs_idx else diacs[i] for i, _ in enumerate(diacs)]

        in_diacs = self._diacritizer.encode(in_diacs)
        in_tokens = self._tokenizer.encode(''.join(twod))
        out_diacs = self._diacritizer.encode(diacs)
        return in_tokens, out_diacs, in_diacs


    def fn_collate(self, batch):
        """Generate a batch of data. For DataLoader's 'collate_fn'.

        Args:
            batch (list(tuple)): A batch of (text inputs, target diacritics, input diacritics).

        """

        inputs = []
        targets = []
        indiacs = []

        for _, data in enumerate(batch):
            inputs.append(data[0])
            targets.append(data[1])
            indiacs.append(data[2])

        inputs_lengths = torch.IntTensor([input.shape[0] for input in inputs ])
        tensor_intokens = pad_sequence(inputs, batch_first=True, padding_value=0)
        # [batch_size, (padded) n_tokens]
        tensor_tardiacs = pad_sequence(targets, batch_first=True, padding_value=-1)
        tensor_indiacs = pad_sequence(indiacs, batch_first=True, padding_value=0)
        return tensor_intokens, tensor_tardiacs, inputs_lengths, tensor_indiacs


if __name__ == "__main__":

    random.seed(10)

    TEXT_FILE = r'C:/Users/Mohammed/my_work/data/tashkeel/aliosm/processed/extended_valid.csv'
    CHAR_VOCAB = r'C:/Users/Mohammed/my_work/data/tashkeel/aliosm/processed/char_vocab.txt'
    DIAC_VOCAB = r'C:/Users/Mohammed/my_work/data/tashkeel/aliosm/processed/diac_vocab.txt'
    PARTIAL_PROB = 0.5

    ttokenizer = Tokenizer(CHAR_VOCAB)
    tdiacritizer = Diacritizer(DIAC_VOCAB, '~')
    tashkeel = Tashkeel(TEXT_FILE, ttokenizer, tdiacritizer, PARTIAL_PROB)
    print('Tokenizer:', ttokenizer.index2char)
    print('Diacritizer:', tdiacritizer.index2diac)
    nb_samples = len(tashkeel)

    samples_idx = np.random.choice(np.arange(nb_samples), replace=False, size=2)

    for i, s in enumerate(samples_idx):
        print('---------------------------------')
        print(f'\nSample {i} index: {s}')
        results = tashkeel[s]
        input_tokens, output_diacs, input_diacs = results

        print('Input Tokens:', input_tokens)
        print('Input diacs:', input_diacs)
        print('output diacs:', output_diacs)

        tokens = ttokenizer.decode(input_tokens.tolist())
        input_diacs = tdiacritizer.decode(input_diacs.tolist())
        target_diacs = tdiacritizer.decode(output_diacs.tolist())

        print(tokens)
        print(target_diacs)
