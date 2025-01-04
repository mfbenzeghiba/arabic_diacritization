# Arabic (Partially) Text diacritization

This repo contains a pytorch implementation of an Arabic Text Diacritization system.
The implementation also, supports partially diacritized text, an arabic text that contains
some words or characters that are already diacritized.
Currently, three variants of RNN-based architectures are supported (e,g; vanilla RNN, LSTM and GRU).
Other types of architectures will be added (e.g, Transformer).

## Dataset

Models are trained on a concatenation of two dataset that are downloded from here [dataset1](https://github.com/AliOsm/arabic-text-diacritization/tree/master/dataset) and here
[dataset2](https://github.com/AliOsm/shakkelha/tree/master/dataset).

The data mainly comes from the Tashkeela database.
The validation dataset composed of the the validation dataset and the test dataset of **dataset1**.

Both train and validation datasets are first preprocessed and cleaned. The cleaning
consists mainly in
1. Split the text into sentences.
2. Remove non-supported characters (non arabic alphabet).
3. Split long sentences.
4. Keep text with high diacritic-to-character ratio rate.

## Training

Training is performed with the .\train_and_eval\train_diacritizer.py script.It takes as argument a YAML config file
that set the data location, model architecture and training hyper-parameters and criterion.

The implementation compares two versions of LSTM models:
1. Baseline model that does not support partially diacritized text as input.
2. Models that do (inspired from the work described in https://arxiv.org/abs/2306.03557)
These models first combine character embeddings and diacritics embeddiungs before feeding them to the RNN model.
Both the sum and the concatenation embeddings rules are implemented.

To train the model with partially diacritized text, a simple technique is used. it consists in the Following steps:
1. For each input sample (a fully diacritized text), a partial probability ((partial_prob (see config file))
is used to decide weither to use partially diacritized text.
2. If so, then a masking probability is randomly chosen to set the percentage of diacritics to be masked (removed)
from the sample.
3. if not, then all diacritics are removed from the sample.

The best model was selected based on the DER (Diacritic Error Rate) computed on the validation set
with fully non diacritized text.

Results show that adding diacritics embeddings slightly improves the performance when the input
text is not diacritized. However the performance can improve significantly when the
input text is partially diacritized.

