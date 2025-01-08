"""Script to evaluate the model."""

from pathlib import Path
import sys
from typing import List, Dict, Tuple
import argparse
import logging

import regex
import numpy as np

import torch
from diacritization_evaluation import util

path_root = Path(__file__).parents[1]
print(path_root)
sys.path.append(str(path_root))

from data_processing.load_data import Tokenizer, Diacritizer
from tools.model_utils import load_checkpoint

def parse_args() ->  argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description='script to train attention models.')
    parser.add_argument('--eval_file', default='',
                        help='Text file to be diacritizede.')
    parser.add_argument('--model_path', default='',
                        help='Text file to be diacritizede.')
    parser.add_argument('--token_vocab', default='',
                        help='The character vocabulary file.')
    parser.add_argument('--diac_vocab', default='',
                        help='The diacritics vocabulary file.')
    parser.add_argument('--model_type', default='RNN',
                        help='The name of the model')
    parser.add_argument('--mask_prob', type=float, default=0.,
                        help='Probabilty of masking diacritics for partial diacritic recognition.')
    parser.add_argument('--out_file', type=str, default='',
                        help='The output file.')

    args = parser.parse_args()
    return args


class Preprocessing():
    """Define a class to preprocess and clean the text."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        super(Preprocessing, self).__init__()
        self.tokenizer = tokenizer
        self.mapping = tokenizer.char2index
        self.char_pos = {}


    def process(self, text: str) -> Tuple[List[str], Dict[int, int]]:
        """Process the input text.

        Args:
            text (str): The input text.

        Returns:
            Tuple[List[str], Dict[int, int]]: a tuple containing the text
            andthe position of non supported characters.
        """

        valid_text = []
        char_pos =  {}
        k = 0

        text = '~'.join(text.split())
        text = regex.sub(u'\u0640', '', text) # strip tatweel
        #text = text.replace('ٱ', 'ا')
        _, twod, in_diacs = util.extract_haraqat(text)

        for i, (c, d) in enumerate(zip(twod, in_diacs)):
            if c in self.mapping:
                valid_text.append(c+d)
                char_pos[i] = k
                k += 1

        return (''.join(valid_text), char_pos)


def main(opts):
    """Eval the given model with the given dataset."""

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    logging.info("-" * 50)
    logging.info('PyTorch Version: %s', torch.__version__)
    logging.info("-" * 50)

    tokenizer = Tokenizer(opts.token_vocab)
    diacritizer = Diacritizer(opts.diac_vocab)
    eval_file = opts.eval_file

    model = load_checkpoint(opts.model_path, opts.model_type.lower())
    model.eval()

    preprocess = Preprocessing(tokenizer)
    res = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            twd = line.strip()
            valid_text, _ = preprocess.process(twd)
            _, vtwod, diacs = util.extract_haraqat(valid_text)

            input_diacs = [''] * len(diacs)
            if opts.mask_prob != 0.:
                nb_masked_diacs = int(opts.mask_prob * len(diacs))
                masked_diacs_idx = np.random.choice(
                    range(len(diacs)), nb_masked_diacs, replace=False
                )
                input_diacs = [
                    '' if i in masked_diacs_idx else diacs[i] for i, _ in enumerate(diacs)
                ]

            inputs = tokenizer.encode(''.join(vtwod)).unsqueeze(0)
            input_diacs = diacritizer.encode(input_diacs).unsqueeze(0)
            ref = []
            with torch.no_grad():
                logits = model(inputs, input_diacs)
                logits = logits.permute(0, 2, 1)
                preds = torch.argmax(logits, dim=1)
                diacs_list = preds[0].tolist()
                hyp_diacs = [diacritizer.index2diac[i] for i in diacs_list ]
                for (c, d) in zip(vtwod, hyp_diacs):
                    if c == '~':
                        c = ' '
                    if d == '~':
                        d = ''
                    ref.append(c+d)
                res.append(''.join(ref))

    with open(opts.out_file, 'w', encoding='utf-8') as f:
        for line in res:
            f.write(f'{line}\n')


if __name__ == "__main__":

    args = parse_args()
    main(args)
