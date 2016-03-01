#!/usr/bin/env python3

from utils import random_round_robin
from texttools import WordEmbedding
from learn import splitsave_XY
import config
from config import resultpath
from result import ClassificationHistogram, ClassificationResult

from itertools import repeat, starmap, islice, chain
import json
import gzip
from copy import copy
from time import time
from collections import namedtuple
from multiprocessing import Process, Queue
from os import makedirs

def iter_ngrams(dataconfig):
    paths = dataconfig.ngrampaths()
    def path_to_ngrams(path):
        f = gzip.open(path, mode='rt')
        for line in f:
            yield line.split()

    return tuple(map(path_to_ngrams, paths))

def iter_XY(dataconfig, runconfig):
    embedding = WordEmbedding(dataconfig.embeddingpath())
    (ngrams, n1grams) = iter_ngrams(dataconfig)

    data = random_round_robin(zip(ngrams, repeat(True)), zip(n1grams, repeat(False)))

    def prepare_gram(gram, isngram):
        if isngram:
            return (embedding[gram], 0)

        l = len(gram)
        assert l % 2 == 1

        middle = l // 2

        if gram[middle] not in runconfig.punctuation:
            return None

        label = runconfig.punctuation.index(gram[middle]) + 1

        skipgram = copy(gram)
        del skipgram[middle]

        return (embedding[skipgram], label)
    
    XY = starmap(prepare_gram, data)
    XY = filter(None, XY)

    return XY

def test_data(XY, label_num, samples_per_label=100):
    label_occ_nums = [0 for _ in range(label_num)]

    # TODO: priors are slightly biased towards rare labels because iteration
    #       stops very likely at a rare label
    #       asymptotically doesn't make a difference though
    class TestDataIterator:
        def done(self):
            return 0 == len([label_occ_num for label_occ_num in label_occ_nums \
                                           if label_occ_num < samples_per_label]) \

        def __iter__(self):
            return self

        def __next__(self):
            if self.done():
                raise StopIteration()

            for x, y in XY:
                label_occ_nums[y] += 1
                if label_occ_nums[y] <= samples_per_label:
                    return (x, y)

            raise BaseException('Not enough training data')

        def priors(self):
            assert self.done()

            total = sum(label_occ_nums)
            return [label_occ_num / total for label_occ_num in label_occ_nums]
    
    return TestDataIterator()
            
def run_classification(dataconfig, runconfig):
    XY = iter_XY(dataconfig, runconfig)

    train_XY = islice(XY, runconfig.train_size)

    start_time = time()
    classifier = runconfig.classify_trainer()(train_XY)
    traintime = round(time() - start_time)

    test_XY = test_data(XY, label_num=len(runconfig.punctuation) + 1)
    test_X, test_Y = splitsave_XY(test_XY)
    priors = test_XY.priors()

    start_time = time()
    response_Y = classifier(test_X)
    testtime = round(time() - start_time)

    def intlabel_to_punctuation(intlabel):
        if intlabel == 0:
            return ''
        else:
            return runconfig.punctuation[intlabel - 1]

    test_Y = map(intlabel_to_punctuation, test_Y)
    response_Y = map(intlabel_to_punctuation, response_Y)

    priors = dict(((intlabel_to_punctuation(intlabel), prior) for (intlabel, prior) in enumerate(priors)))

    histogram = ClassificationHistogram.from_labels(chain([''], runconfig.punctuation))
    for test_y, response_y in zip(test_Y, response_Y):
        bin_index = next((i for (i, c) in enumerate(histogram) if c.truth == test_y
                                                               if c.prediction == response_y))
        b = histogram[bin_index]
        histogram[bin_index] = b._replace(number=b.number + 1)

    return ClassificationResult( traintime=traintime \
                               , testtime=testtime \
                               , priors=priors \
                               , histogram=histogram )

def main(dataconfigs=config.dataconfigs, runconfigs=config.runconfigs, timeout=10 * 60):
    makedirs('data/results', exist_ok=True)
    for dataconfig in dataconfigs:
       for runconfig in runconfigs:
           out_queue = Queue()
           def worker():
               out_queue.put(run_classification(dataconfig, runconfig))
   
           p = Process(target=worker)
           p.start()
           p.join(timeout)
           if not p.is_alive():
               analysis = out_queue.get()
               outpath = resultpath(dataconfig, runconfig)
               out_jso = { 'data': dataconfig.to_jso() \
                         , 'runconfig': runconfig.to_jso() \
                         , 'result': analysis.to_jso() }
               with gzip.open(outpath, mode='xt') as f:
                   print(json.dumps(out_jso, indent=4), file=f)
           else:
               print('Timeout')

if __name__ == '__main__':
    main()
