#!/usr/bin/python

import time
import numpy as np
import codecs
import argparse
import math
import sys

# Initializing the logging module
import logging
import log_utils as L

# For feature functions

# For KenLM features
sys.path.insert(0, 'lib/kenlm_python/')
import kenlm


# For edit operations feature
from lib.m2scorer_optimized import levenshtein

logger = logging.getLogger(__name__)

ln10 = math.log(10)
class LM:
    def __init__(self, name, path, normalize=False, debpe=False):
        self.path = path
        c = kenlm.Config()
        c.load_method = kenlm.LoadMethod.LAZY
        self.model = kenlm.Model(path, c)
        self.name = name
        self.normalize = normalize
        self.debpe = debpe
        logger.info('Intialized ' + str(self.model.order) + "-gram language model: " + path)

    def get_name(self):
        return self.name

    def get_score(self, source, candidate):
        if self.debpe:
            candidate = candidate.replace('@@ ','')
        lm_score = self.model.score(candidate)
        log_scaled = round(lm_score*ln10,4)
        if self.normalize == True:
            if len(candidate):
                return (log_scaled * 1.0 ) / len(candidate.split())
        return str(round(lm_score*ln10,4))

class SAMPLE:
    def __init__(self, name):
        self.name = name

    def get_score(self, source, candidate):
        return str(0.5)


class WordPenalty:
    '''
        Feature to caclulate word penalty, i.e. number of words in the hypothesis x -1
    '''
    def __init__(self, name):
        self.name = name

    def get_score(self, source, candidate):
        return str(-1 * len(candidate.split()))

class EditOps:
    '''
        Feature to calculate edit operations, i.e. number of deletions, insertions and substitutions
    '''
    def __init__(self, name, dels=True, ins=True, subs=True):
        self.name = name
        self.dels = ins
        self.ins = ins
        self.subs = subs

    def get_score(self, source, candidate):
        src_tokens = source.split()
        trg_tokens = candidate.split()
        # Get levenshtein matrix
        lmatrix, bpointers = levenshtein.levenshtein_matrix(src_tokens, trg_tokens, 1, 1, 1)

        r_idx = len(lmatrix)-1
        c_idx = len(lmatrix[0])-1
        ld = lmatrix[r_idx][c_idx]
        d = 0 
        i = 0 
        s = 0 
        bpointers_sorted = dict()

        for k, v in bpointers.iteritems():
            bpointers_sorted[k] =sorted(v, key=lambda x: x[1][0])

        # Traverse the backpointer graph to get the edit ops counts
        while (r_idx != 0 or c_idx != 0): 
            edit = bpointers_sorted[(r_idx,c_idx)][0]
            if edit[1][0] == 'sub':
                s = s+1
            elif edit[1][0] == 'ins':
                i = i+1
            elif edit[1][0] == 'del':
                d = d+1
            r_idx = edit[0][0]
            c_idx = edit[0][1]
        scores = ""
        if self.dels:
            scores += " " + str(d) 
        if self.ins:
            scores += " " + str(i) 
        if self.subs:
            scores += " " + str(s)
        return scores

class LexWeights:
    '''
    Use translation model from SMT p(w_f|w_e) using the alignment model
    '''
    def __init__(self, name, align_file):

        with open(align_file) as f:
            for line in f:
                print line.strip()
    


