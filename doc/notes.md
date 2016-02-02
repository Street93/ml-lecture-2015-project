# Scrape sample data

We need to get some high-quality samples of German (maybe English) text in a consistent format.
Sources for German text are

* https://www.gutenberg.org/
* http://www.spiegel.de/spiegel/print/
* http://www.zeit.de/2015/index
* http://fazarchiv.faz.net/
* https://wikipedia.de/

We have to be careful and not mix text that was written against different official spellings, e.g. text from before and after the spelling reform of 1996.
It may be interesting to see whether we can see the spelling reform when learning from older or newer text and comparing the respective distributions.


# Creating a word embedding

## Google's [word2vec](https://code.google.com/archive/p/word2vec/)

This is an algorithm based on a 2013 [paper](http://arxiv.org/abs/1310.4546) by some Google engineers.
This is probably easy to use, but we should probably also try to build our own word embedding.

## Use [TensorFlow](https://www.tensorflow.org/) to roll our own word embedding

It's probably not easy to beat word2vec, but maybe we can come close to it using a high level library for neural networks like TensorFlow.

## Use primitives from numpy or scipy

Probably not easy: Neural networks are hard to implement efficiently, so we can only use simpler approaches like something based on co-occurence matrices.


# Using geometry of word embeddings for prediction

We may use word embeddings like this:
For a given gap between words, let X be vector obtained by concatenating the word embeddings of the nearest k words for some k.
Based on X, try to predict whether we need to insert a comma, period or some other character.
Word embeddings typically have the property that similar words are close to each other.
We may use this to predict based on the geometry of the distribution of X.
We may also use variations on this theme:
What happens if, instead of concatenating the word vectors of the nearest k neighboring words, we add them?
How important is the actual word order, i.e. can we permute the nearest k neighbors before putting them together into X?
You'd expect to see a decrease of importance of a word the further it is a away from the current gap.
Can we see this in our classifier?
What is a good value for k?
How does English text differ from German one in these respects?


Algorithms we may use for this:

* Nearest neighbor classifier  
  Probably the simplest approach.
* LDM/QDM  
  I don't see why this should work very well, but let's try.
* Least squares  
  Seems to be robust for many scenarios.
* Try to fit some other geometry
  Can we fit some other shapes on our data?
  Spheres? Polyedra? 
* Deep neural networks  
  I don't know much about this, but people use this and it seems to work well for other problems.


# Other algorithms

I can't really think of an algorithm that doesn't use word embeddings...
Algorithms won't have already seen most phrases occuring in the test data in the training data, so we need some kind of 'distance' between phrases, which naturally leads to word embeddings.
Some algorithms that come to mind that are not obviously 'geometric' or do  not even use word embeddings:


* Markov chain based method  
  This is often used for spelling correction.
  How well does this perform when used for punctuation correction?
  As markov chains only consider words only to the left or only to the right, this will probably yield worse results than the other algorithms.
  Due its online nature with obvious advantages, this may be interesting to look into nevertheless.
* Use a nearest neighbor classifier *without* word embeddings  
  Only look at the nearest k words with some very small k, so most observations occure exactly in training data.
  If a phrase is encountered that doesn't occure in training data, do nothing.
  Very crude, but easy to implement.
* Investigate whether there are algorithms for constructing a parser based on data  
  Try to learn actual German/English grammar and build a syntax tree, then predict based on the tree whether to insert some punctuation mark.
  Seems very hard or even impossible (can we construct a robust embedding into a tree structure?).

