from utils import to_ndarray, Struct

from collections import namedtuple
from numpy import array, ones, zeros, concatenate, dot, fromiter, float32, \
                  cov as npcov, mean as npmean, log, ndarray
from numpy.linalg import norm, lstsq
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.spatial.distance import cosine
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from functools import partial

def train_nn_classifier(yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)

    samples = concatenate((yes_samples, no_samples))

    nn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    if len(samples.shape) == 1:
        nn = nn.fit(samples.reshape((-1, 1)))
    else:
        nn = nn.fit(samples)
 
    def classify(sample):
        sample = to_ndarray(sample)
        if type(sample) == ndarray:
            sample = sample.reshape((1, -1))
        neighbor_indices = nn.kneighbors(sample, return_distance=False)
        yes_neighbor_indices = [index[0] for index in neighbor_indices if index[0] < len(yes_samples)]
        no_neighbor_indices = [index[0] for index in neighbor_indices if index[0] >= len(yes_samples)]
        
        if len(yes_neighbor_indices) > len(no_neighbor_indices):
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

def bayesian_classifier( yes_pdf=None \
                       , no_pdf=None \
                       , yes_logpdf=None \
                       , no_logpdf=None \
                       , yes_prior=None ):
    assert yes_prior is not None
    no_prior = 1 - yes_prior


    if yes_pdf is None:
        assert no_pdf is None
        assert yes_logpdf is not None
        assert no_logpdf is not None
        yes_logprior = log(yes_prior)
        no_logprior = log(no_prior)

        def classify(sample):
            sample = to_ndarray(sample)
            yes_logprob = yes_logpdf(sample) + yes_logprior
            no_logprob = no_logpdf(sample) + no_logprior

            if yes_logprob > no_logprob:
                return True
            else:
                return False

        return classify

    else:
        assert no_pdf is not None
        assert yes_logpdf is None
        assert no_logpdf is None

        def classify(sample):
            sample = to_ndarray(sample)
            yes_prob = yes_pdf(sample) + yes_prior
            no_prob = no_pdf(sample) + no_prior

            if yes_prob > no_prob:
                return True
            else:
                return False

        return classify

def train_normal_dist(samples):
    mean = npmean(samples, axis=0)
    cov = npcov(samples, rowvar=0)

    pdf = lambda x: multivariate_normal.pdf(x, mean=mean, cov=cov) # , allow_singular=True)
    logpdf = lambda x: multivariate_normal.logpdf(x, mean=mean, cov=cov) #, allow_singular=True)

    return Struct(pdf=pdf, logpdf=logpdf, mean=mean, cov=cov)

def train_qda_classifier(yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)

    yes_dist = train_normal_dist(yes_samples)
    no_dist = train_normal_dist(no_samples)
    yes_prior = len(yes_samples) / (len(yes_samples) + len(no_samples))

    return bayesian_classifier( yes_logpdf=yes_dist.logpdf \
                              , no_logpdf=no_dist.logpdf \
                              , yes_prior=yes_prior )

def train_dimred_classifier(train_dimred, train_classifier, yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)

    reduce_dim = train_dimred(yes_samples, no_samples)
    yes_samples = map(reduce_dim, yes_samples)
    no_samples = map(reduce_dim, no_samples)

    classify_reduced = train_classifier(yes_samples, no_samples)

    classify = lambda sample: classify_reduced(reduce_dim(sample))

    return classify

train_lstsq_qda_classifier = partial( train_dimred_classifier \
                                    , train_lstsq_dimred \
                                    , train_qda_classifier )

train_lstsq_nn_classifier = partial( train_dimred_classifier \
                                   , train_lstsq_dimred \
                                   , train_nn_classifier )

def train_gaussian_kde_classifier(yes_samples, no_samples, bw_method='scott'):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)

    yes_pdf_est = gaussian_kde(yes_samples.T, bw_method)
    no_pdf_est = gaussian_kde(no_samples.T, bw_method)

    yes_logpdf = lambda x: yes_pdf_est.logpdf(x)[0]
    no_logpdf = lambda x: no_pdf_est.logpdf(x)[0]
    yes_prior = len(yes_samples) / (len(yes_samples) + len(no_samples))

    return bayesian_classifier( yes_logpdf=yes_logpdf \
                              , no_logpdf=no_logpdf \
                              , yes_prior=yes_prior )

def train_tophat_kde_classifier(yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)

    yes_pdf_est = KernelDensity(kernel='tophat', bandwidth=0.9).fit(yes_samples)
    no_pdf_est = KernelDensity(kernel='tophat', bandwidth=0.9).fit(no_samples)

    yes_logpdf = lambda x: yes_pdf_est.score(x.reshape(1, -1))
    no_logpdf = lambda x: no_pdf_est.score(x.reshape(1, -1))
    yes_prior = len(yes_samples) / (len(yes_samples) + len(no_samples))

    return bayesian_classifier( yes_logpdf=yes_logpdf \
                              , no_logpdf=no_logpdf \
                              , yes_prior=yes_prior )

def train_random_forest_classifier(yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples)
    no_samples = to_ndarray(no_samples)

    yes_labels = ones(len(yes_samples))
    no_labels = zeros(len(no_samples))

    samples = concatenate((yes_samples, no_samples))
    labels = concatenate((yes_labels, no_labels))

    if len(samples.shape) == 1:
        rnd_tree = RandomForestClassifier().fit(samples.reshape((-1, 1)), labels)

        def classify(sample):
            if rnd_tree.predict(to_ndarray(sample))[0] == 1:
                return True
            else:
                return False

        return classify
    else:
        rnd_tree = RandomForestClassifier().fit(samples, labels)

        def classify(sample):
            if rnd_tree.predict(to_ndarray(sample).reshape((1, -1)))[0] == 1:
                return True
            else:
                return False

        return classify

    

train_lstsq_random_forest_classifier = partial( train_dimred_classifier \
                                              , train_lstsq_dimred \
                                              , train_random_forest_classifier )
