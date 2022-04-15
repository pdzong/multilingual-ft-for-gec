#!/bin/bash

# apt-get update
# apt-get -y upgrade
# apt-get install rsync mc screen git htop

pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/huggingface/transformers.git
cd transformers/
git checkout 40ecaf0c2b1c0b3894e9abf619f32472c5a3b3ca
cp ../finetune.py ./examples/seq2seq/
cp ../utils.py ./examples/seq2seq/
cp ../tokenization_xlm_prophetnet.py ./src/transformers/models/xlm_prophetnet/
cp ../tokenization_mbart.py ./src/transformers/models/mbart/
cp ../tokenization_mbart_fast.py ./src/transformers/models/mbart/
cp ../modeling_mbart.py ./src/transformers/models/mbart/
cp ../modeling_mt5.py ./src/transformers/models/mt5/
pip install --editable .
pip install jieba
pip install sentencepiece==0.1.85
pip install pytorch_lightning==1.0.8
pip install sacremoses
pip install spacy==2.3.0
python -m spacy download en_core_web_sm
pip install gitpython
pip install rouge_score
pip install sacrebleu
pip install nltk