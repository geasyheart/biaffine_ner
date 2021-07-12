# -*- coding: utf8 -*-
#
from src import biaffine_ner
from src.utils import TEST_FILE

bn = biaffine_ner.BiaffineNer()
bn.predict(
    file=TEST_FILE,
    transformer='hfl/chinese-electra-180g-small-discriminator',
    sequence_length=128
)
