# Arabic (Partially) Text diacritization

This repo contains a pytorch implementation of an Arabic Text Diacritization system.
The implementation also, supports partially diacritized text, an arabic text that contains
some words or characters that are already diacritized.
Currently, three variants of RNN-based architectures are supported. These are vanilla RNN, LSTM and GRU.
Other types of architectures will be added (e.g, Transformer).

## Dataset

Models are trained on a concatenation of two datasets that are downloded from here [`dataset1`](https://github.com/AliOsm/arabic-text-diacritization/tree/master/dataset) and here
[`dataset2`](https://github.com/AliOsm/shakkelha/tree/master/dataset). The validation dataset composed of the validation and the test datasets of `dataset1`. The data mainly comes from the Tashkeela corpus.

Both train and validation datasets are first preprocessed and cleaned. The cleaning consists mainly in:
1. Split the text into sentences.
2. Remove non-supported characters (non arabic alphabet).
3. Split long sentences.
4. Keep text with high diacritic-to-character ratio rate.

## Training

Training is performed with the `train_and_eval\train_diacritizer.py` script. It takes as argument a YAML configuration file
that set the data location, model architecture and training hyper-parameters and criterion.

The implementation compares two versions of RNN-based models:
1. Baseline model that does not support partially diacritized text as input.
2. Models that do (inspired from the work described in https://arxiv.org/abs/2306.03557). These models first combine character embeddings and diacritic embeddings before feeding them to the RNN model.
Both the sum and the concatenation embedding rules are implemented.

To train the model with partially diacritized text, a simple technique is used. it consists in the Following steps:
1. For each input sample (a fully diacritized text), a partial probability (partial_prob option in the config file) is used to decide weither to use partially diacritized text or not.
2. If so, then a masking probability is randomly chosen to set the percentage of diacritics to be masked (removed) from the sample.
3. if not, then all diacritics are removed from the sample.

The best model was selected based on the DER (Diacritic Error Rate) computed on the validation set with fully non diacritized text.

Results show that combining character and diacritic embeddings slightly improves the performance when the input text is not diacritized. However the performance can improve significantly when the
input text is partially diacritized.

## Gradio App

A gradio prototype using a LSTM model is developed. The app is hosted in HuggingFace space.
It can be accessed through this link [arabic\_diacritizer](https://huggingface.co/spaces/benmfb/arabic_diacritizer).

The application has the Following characteristics:
1. The app accepts any kind of text without any constraint.
2. It keeps non Arabic text as it is.
3. It supports partially diacritized text, it keeps the diacritized text as it is and adds diacritics to the remaining text.
4. The user can enter a text (manually or copy/past) or upload a text file.
5. The app allowed copy-pasting the diacritized text to be saved in a file.
