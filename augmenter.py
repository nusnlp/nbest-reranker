#!/usr/bin/python

import time
from candidatesreader import NBestList
import numpy as np
import codecs
import argparse

import os
import logging
import log_utils as L
from features import *

logger = logging.getLogger(__name__)


def augment(feature, source_path, input_nbest_path, output_nbest_path):
    # Initialize NBestList objects
    logger.info('Initializing Nbest lists')
    input_nbest = NBestList(input_nbest_path, mode='r')
    output_nbest = NBestList(output_nbest_path, mode='w')

    # Load the source sentences
    logger.info('Loading source sentences')
    src_sents = codecs.open(source_path, mode='r', encoding='UTF-8')
    
    # For each of the item in the n-best list, append the feature
    sent_count = 0
    for group, src_sent in zip(input_nbest, src_sents):
        for item in group:
            item.append_feature(feature.get_score(src_sent, item.hyp))
            output_nbest.write(item)
        sent_count += 1
        if (sent_count % 100 == 0):
            logger.info('Augmented ' + L.b_yellow(str(sent_count)) + ' sentences.')
    output_nbest.close()
    logger.info(L.green('Augmenting done.'))


parser = argparse.ArgumentParser()	
parser.add_argument("-s", "--source-sentence-file", dest="source_path", required=True, help="Source sentnece file")
parser.add_argument("-i", "--input-nbest", dest="input_nbest_path", required=True, help="Input n-best file")
parser.add_argument("-o", "--output-nbest", dest="output_nbest_path", required=True, help="Output n-best file")
parser.add_argument("-f", "--feature", dest="feature_string", required=True, help="feature initializer, e.g. kenlm.LM('LM0','/path/to/lm_file')")
args = parser.parse_args()

L.set_logger(os.path.abspath(os.path.dirname(args.output_nbest_path)),'augment_log.txt')
L.print_args(args)
feature = eval(args.feature_string)
augment(feature, args.source_path, args.input_nbest_path, args.output_nbest_path)
