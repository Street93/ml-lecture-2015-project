from utils import compress, decompress

import numpy as np
import os
import gzip
import subprocess
from pathlib import PurePath
import random
import string

def create_word_embedding(infile, outfile, min_count=5, size=100, downsample=True, estimator='skipgram', negative=5):
    def run_on(actual_infile):
        assert estimator in ['skipgram', 'cbow']
        use_cbow = estimator == 'cbow'
        sample = 10e-3
        if not downsample:
            sample = 1
        subprocess.run([ 'tools/word2vec/word2vec' \
                       , '-train', actual_infile \
                       , '-output', outfile \
                       , '-min-count', str(min_count) \
                       , '-size', str(size) \
                       , '-sample', str(downsample) \
                       , '-negative', str(negative) \
                       , '-cbow', str(int(use_cbow)) ])

    compress_flag = False
    outfilep = PurePath(outfile)
    if outfilep.suffix == '.gz':
        outfile = str(outfilep.parent / outfilep.stem)
        compress_flag = True
    
    infilep = PurePath(infile)
    if infilep.suffix == '.gz':
        rndstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
        tmp_filename = '/tmp/' + rndstr
        try:
            decompress(infile, keep=True, outfile=tmp_filename)
            run_on(tmp_filename)
        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)
    else:
        run_on(infile)

    if compress_flag:
        compress(outfile)

class WordEmbedding:
    def __init__(self, file_name):
        f = gzip.open(file_name, mode='rt')

        firstline = f.readline()
        [word_num_str, dimension_str] = firstline.split()
        word_num = int(word_num_str)
        dimension = int(dimension_str)

        self.word_indices = {}
        self.values = np.zeros((word_num, dimension))

        word_index = 0
        for line in f:
            items = line.split()

            word = items[0]
            self.word_indices[word] = word_index

            str_vec = items[1 : ]
            assert len(str_vec) == dimension
            for i in range(0, dimension):
                self.values[word_index, i] = float(str_vec[i])

            word_index += 1
        
        assert len(self.word_indices) == word_num

    def __getitem__(self, param):
        return self.values[self.word_indices[param]]

def ngram_to_vec(ngram, embedding):
    return list(chain.from_iterable(map(embedding.__getitem__, ngram)))
