from utils import to_ndarray, Struct

from collections import namedtuple
from numpy import array, ones, zeros, concatenate, dot, fromiter, float32, \
                  cov as npcov, mean as npmean
from numpy.linalg import norm, lstsq
from scipy.stats import multivariate_normal, gaussian_kde

def train_nn_classifier(yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)
    
    def classify(sample):
        sample = to_ndarray(sample)
        dist = lambda s: norm(s - sample)
        yes_dists = map(dist, yes_samples)
        no_dists = map(dist, no_samples)

        if min(yes_dists) > min(no_dists):
            return True
        else:
            return False

    return classify

def train_lstsq_dimred(yes_samples, no_samples):
    yes_a = to_ndarray(yes_samples)
    yes_b = ones(yes_a.shape[0])

    no_a = to_ndarray(no_samples)
    no_b = -ones(no_a.shape[0])

    a = concatenate((yes_a, no_a))
    b = concatenate((yes_b, no_b))

    (coeff_vec, _, _, _) = lstsq(a, b)

    dimred = lambda sample: dot(coeff_vec, to_ndarray(sample))

    return dimred

def bayesian_classifier(yes_pdf, no_pdf, yes_prior):
    no_prior = 1 - yes_prior
    
    def classify(sample):
        sample = to_ndarray(sample)
        yes_prob = yes_pdf(sample) * yes_prior
        no_prob = no_pdf(sample) * no_prior
        if yes_prob > no_prob:
            return True
        else:
            return False

    return classify

def train_normal_dist(samples):
    mean = npmean(samples, axis=0)
    cov = npcov(samples, rowvar=0)

    pdf = lambda x: multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=True)
    logpdf = lambda x: multivariate_normal.logpdf(x, mean=mean, cov=cov, allow_singular=True)

    return Struct(pdf=pdf, logpdf=logpdf, mean=mean, cov=cov)

def train_qda_classifier(yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)

    yes_dist = train_normal_dist(yes_samples)
    no_dist = train_normal_dist(no_samples)
    yes_prior = len(yes_samples) / (len(yes_samples) + len(no_samples))

    return bayesian_classifier(yes_dist.pdf, no_dist.pdf, yes_prior)

def train_dimred_classifier(train_dimred, train_classifier, yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)

    reduce_dim = train_dimred(yes_samples, no_samples)
    yes_samples = map(reduce_dim, yes_samples)
    no_samples = map(reduce_dim, no_samples)

    classify_reduced = train_classifier(yes_samples, no_samples)

    classify = lambda sample: classify_reduced(reduce_dim(sample))

    return classify

def train_lstsq_qda_classifier(yes_samples, no_samples):
    return train_dimred_classifier( train_dimred=train_lstsq_dimred \
                                  , train_classifier=train_qda_classifier \
                                  , yes_samples=yes_samples \
                                  , no_samples=no_samples )

def train_gaussian_kde_classifier(yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)

    yes_pdf_est = gaussian_kde(yes_samples.T)
    no_pdf_est = gaussian_kde(no_samples.T)

    yes_pdf = lambda x: yes_pdf_est(x)[0]
    no_pdf = lambda x: no_pdf_est(x)[0]
    yes_prior = len(yes_samples) / (len(yes_samples) + len(no_samples))

    return bayesian_classifier(yes_pdf, no_pdf, yes_prior)
