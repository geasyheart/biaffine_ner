# -*- coding: utf8 -*-
#

from src import biaffine_ner
from src.utils import TRAIN_FILE, TEST_FILE

bn = biaffine_ner.BiaffineNer()
bn.fit(
    train_data=TRAIN_FILE,
    dev_data=TEST_FILE,
    transformer='ckiplab/albert-tiny-chinese',

)
