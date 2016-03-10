# torch-lm
torch-lm is a Torch library for training language models.

It takes any UTF-8 input and learns a character-aware language model. The input is split using regular expressions into sentences and then into words (cut off at 25 characters). Words are encoded as tokens, represented as fixed-size vectors. No further pre-processing is performed.

The project is currently WIP. The goal is to develop a general-purpose library for:

- Sentence identification
- Sentiment analysis
- Spelling correction
- Grammar checking

## Installation
Make sure that Torch is installed. Additionally, the following dependencies need to be installed:

```bash
luarocks install --server=http://luarocks.org/dev fun
luarocks install lrexlib-pcre
luarocks install class
luarocks install luautf8
luarocks install rnn
luarocks install nngraph
```

For CUDA support:

```bash
luarocks install cunn
```

For OpenCL support:

```bash
luarocks install clnn
```

## Training
Save the text corpus as `input.txt`. To train the model, run:

```bash
th Model.lua
```

## Credits
The code is largely inspired by [lstm-char-cnn](https://github.com/yoonkim/lstm-char-cnn), but was redesigned with FP principles in mind. toch-lm uses the [`rnn` Torch module](https://github.com/Element-Research/rnn) which provides an implementation for LSTM.

## License
torch-lm is licensed under the terms of the Apache v2.0 license.

## Authors
* Tim Nieradzik
