#!/usr/bin/env python3

import spiegel
from spiegel import load_raw_articles, SpiegelIssue, IssueDoesNotExist
from sanitize import sanitize_article
from utils import retrying
from texttools import create_word_embedding, WordEmbedding, random_corpus_ngrams, \
                      create_vocabulary, load_vocabulary
import gzip
from numpy import random
from os import makedirs
import requests
from itertools import islice
from pathlib import Path

def download_spiegel(first_issue=SpiegelIssue(1970, 1)):
    makedirs('data/spiegel', exist_ok=True)
    issues = (SpiegelIssue(year, week) for year in range(first_issue.year, 2010) \
                                       for week in range(first_issue.week, 53))

    for issue in issues:
        @retrying( requests.exceptions.ConnectionError \
                 , retries=1 \
                 , retry_delay=120 )
        def retrying_load():
            return load_raw_articles(issue, topath='data/spiegel')

        try:
            retrying_load()
        except IssueDoesNotExist:
            pass
        except FileExistsError:
            pass
        except requests.exceptions.ConnectionError as exc:
            print('Failed to download issue {}-{:02}'.format(year, week), file=stderr)
            print(exc, file=stderr)

def create_corpora():
    makedirs('data/copora', exist_ok=True)
    with gzip.open('data/corpora/spiegel-full.gz', mode='wt') as f:
        articles = map(sanitize_article, spiegel.iter_issues_articles())
        for article in articles:
            for paragraph in article:
                print(paragraph, file=f)

def create_embeddings():
    makedirs('data/embeddings', exist_ok=True)
    for corpuspath in Path('data/corpora').iterdir():
        name1 = corpuspath.stem
        for size in [10, 50, 100, 500, 1000]:
            name2 = name1 + '-' + str(size)
            for downsample in [True, False]:
                name3 = name2 + '-' + str(downsample)
                for estimator in ['skipgram', 'cbow']:
                    name4 = name3 + '-' + estimator
                    for negative in [5, 15]:
                        name5 = name4 + '-' + str(negative)

                        outpath = Path('data/embeddings') / (name5 + '.gz')
                        create_word_embedding( infile=str(corpuspath) \
                                             , outfile=str(outpath) \
                                             , size=size \
                                             , downsample=downsample \
                                             , estimator=estimator \
                                             , negative=negative )

def create_vocabularies():
    makedirs('data/vocabularies', exist_ok=True)
    for corpuspath in Path('data/corpora').iterdir():
        corpusname = corpuspath.stem
        vocabfile = 'data/vocabularies/{}.gz'.format(corpusname)
        vocab = create_vocabulary(str(corpuspath), vocabfile)

def create_ngrams():
    makedirs('data/ngrams', exist_ok=True)
    for corpuspath in Path('data/corpora').iterdir():
        corpusname = corpuspath.stem
        vocab = load_vocabulary('data/vocabularies/' + corpusname + '.gz')
        def valid_ngram(ngram):
            return all(map(lambda word: word in vocab, ngram))
        for N in [4, 5, 10, 11]:
            ngrams = random_corpus_ngrams( str(corpuspath) \
                                         , N \
                                         , number=10000 \
                                         , predicate=valid_ngram )
            outfile = 'data/ngrams/{}-{}-grams.gz'.format(corpusname, N)
            with gzip.open(outfile, mode='xt') as f:
                for ngram in ngrams:
                    print(' '.join(ngram), file=f)

def main():
    print('Downloading Spiegel issues... ')
    download_spiegel()
    print('Done.')
    print('Scraping text corpora... ')
    create_corpora()
    print('Done.')
    print('Creating text embeddings... ')
    create_embeddings()
    print('Done.')
    print('Extracting ngrams... ')
    create_ngrams()
    print('Done.')

if __name__ == '__main__':
    main()
