    ######################################## Prepared for the unsupervised machine translation (bilingual part, various ways to get the initial translation) ########################################
    ######################################## 2: phrase level translation with Moses ########################################

import argparse
import torch
import numpy as np
import random

parser = argparse.ArgumentParser(description="preprocess.py")

# the phrase here is the aligned phrase embedding in the same space

parser.add_argument("-dict", type=str,default="/share/data/speech/Data/haiwang/projects/ttic_nlp/MUSE-master2/MUSE/data/crosslingual/dictionaries/en-fr.txt",
                    help="Path to the source transformed embedding comes from src to target direction")

# output, for the phrase table, we can use this to pre-training the encoder-decoder
# output this with two formats
parser.add_argument("-phrasetable", default="en-fr",
                    help="save the phrase table from src to tgt")

opt = parser.parse_args()


def read_dict(path):
    """
    Read all paraphrase from a word embedding file, the embeddings must contains all the tokens that apprear in the training set
    , we can use obtain this from the fastext subword information. 
    """
    bidict = []
    cnt = 0
    with open(path, 'r') as f:
        line = f.readline()
        for line in f:
            # first use the " " to test the code
            word1, word2 = line.split(' ') # the embedding are seperated by the "\t", not by ""
            bidict.append((word1, word2.strip()))
            
    return bidict

def write_phrasetable_moses(file_name, bidict):
    
    fp = open(file_name, "wt+")
    # we can only choose the top k
    for item in bidict:
        # some phrase has "_", replace it with "" and trim
        fp.write(item[0] + " ||| " + item[1] + " ||| " + "1.0" + " ||| |||\n")
    fp.close()

def construct_translation(opt):

    bi_dict = read_dict(opt.dict)
    write_phrasetable_moses(opt.phrasetable, bi_dict)

def main():
    construct_translation(opt)
if __name__ == "__main__":
    main()
