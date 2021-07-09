# -*- coding: utf8 -*-
#
from src import biaffine_ner
from src.utils import TEST_FILE

bn = biaffine_ner.BiaffineNer()
bn.predict(
    file=TEST_FILE,
    transformer='bert-base-chinese',
)
