#!/usr/bin/python

import time
import numpy as np
import codecs
import argparse
import math
import sys

sys.path.insert(0, 'lib/kenlm_python/')
import kenlm

ln10 = math.log(10)

class LM:
    def __init__(self, feature_name, path, normalize=False):
        self.path = path
        c = kenlm.Config()
        c.load_method = kenlm.LoadMethod.LAZY
        self.model = kenlm.Model(path, c)
        self.name = feature_name
        self.normalize = normalize
        print >> sys.stderr,  str(self.model.order) + "-gram language model"

    def get_name(self):
        return self.name

    def get_score(self, source, candidate):
        lm_score = self.model.score(candidate)
        log_scaled = round(lm_score*ln10,4)
        if self.normalize == True:
            if len(candidate):
                return (log_scaled * 1.0 ) / len(candidate.split())
        return round(lm_score*ln10,4)


