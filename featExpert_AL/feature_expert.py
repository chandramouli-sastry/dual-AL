# FILE taken from https://github.com/iitml/ALwR
import sys
import os

sys.path.append(os.path.abspath("."))

from time import time
import numpy as np
import scipy.sparse as sp
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
import warnings

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)


class feature_expert(object):
    '''
    feature expert returns what it deems to be the most informative feature
    given a document

    feature expert ranks the features using one of the following criteria:
        1. mutual information
        2. logistic regression with L1 regularization weights
        3. Chi squared statistic
    '''

    def __init__(self, X, y, metric, smoothing=1e-6, C=0.1, seed=12345, pick_only_top=False):
        self.sample_size, self.num_features = X.shape
        self.metric = metric
        self.smoothing = smoothing
        self.feature_rank = ([], [])
        self.seed = seed
        self.rg = np.random.RandomState(seed)
        self.class1_prob = np.sum(y) / float(len(y))

        print '-' * 50
        print 'Starting Feature Expert Training ...'
        start = time()

        if metric == 'mutual_info':
            self.feature_rank = self.rank_by_mutual_information(X, y)
        elif metric == 'chi2':
            self.feature_rank = self.rank_by_chi2(X, y)
        elif metric == 'L1':
            self.feature_rank = self.L1_rank(C, X, y)
        elif metric == 'L1-count':
            self.feature_rank = self.rank_by_L1_weights(C, X, y)
        else:
            raise ValueError('metric must be one of the following: \'mutual_info\', \'chi2\', \'L1\', \'L1-count\'')

        if pick_only_top:

            num_inst, num_feat = X.shape

            the_top = np.zeros(shape=(2, num_feat))

            for i in range(num_inst):
                mif = self.most_informative_feature(X[i], y[i])
                if mif:
                    the_top[y[i]][mif] += 1

            c0_frequency = np.diff(X[np.nonzero(y == 0)[0]].tocsc().indptr)
            c1_frequency = np.diff(X[np.nonzero(y == 1)[0]].tocsc().indptr)

            frequency = (c0_frequency, c1_frequency)

            include_feats = set()

            min_percent = 0.05

            for f in range(num_feat):
                top_freq = 0

                if the_top[0][f] > 0:
                    top_freq = the_top[0][f] / (float(frequency[0][f] + 0.001))
                else:
                    top_freq = the_top[1][f] / (float(frequency[1][f] + 0.001))

                if top_freq >= min_percent:
                    include_feats.add(f)

            new_class0_feats = []
            new_class1_feats = []

            for f in self.feature_rank[0]:
                if f in include_feats:
                    new_class0_feats.append(f)

            for f in self.feature_rank[1]:
                if f in include_feats:
                    new_class1_feats.append(f)

            self.feature_rank = (new_class0_feats, new_class1_feats)

        print 'Feature Expert has deemed %d words to be of label 0' % len(self.feature_rank[0])
        print 'Feature Expert has deemed %d words to be of label 1' % len(self.feature_rank[1])

        print 'Feature Expert trained in %0.2fs' % (time() - start)

    def class0_features_by_rank(self):
        return self.feature_rank[0]

    def class1_features_by_rank(self):
        return self.feature_rank[1]

    def info(self):
        print 'feature expert is trained from %d samples on % features using metric \'%s\'' % \
              (self.sample_size, self.num_features, self.metric)

    def rank_by_mutual_information(self, X, y):
        self.feature_count = self.count_features(X, y)
        self.feature_mi_scores = np.zeros(shape=self.num_features)

        for f in range(self.num_features):
            probs = self.feature_count[f] / self.feature_count[f].sum()
            f_probs = probs.sum(1)
            y_probs = probs.sum(0)

            for i in range(2):
                for j in range(2):
                    self.feature_mi_scores[f] += probs[i, j] * (np.log2(probs[i, j]) - np.log2(f_probs[i])
                                                                - np.log2(y_probs[j]))

        self.feature_scores = self.feature_mi_scores

        feature_rank = np.argsort(self.feature_mi_scores)[::-1]

        return self.classify_features(feature_rank)

    def rank_by_chi2(self, X, y):

        self.feature_count = self.count_features(X, y)

        chi2_scores = chi2(X, y)

        self.feature_scores = chi2_scores[0]

        nan_entries = np.nonzero(np.isnan(self.feature_scores))

        self.feature_scores[nan_entries] = 0

        feature_rank = np.argsort(self.feature_scores)[::-1]

        return self.classify_features(feature_rank)

    def L1_rank(self, C, X, y):
        clf_l1 = linear_model.LogisticRegression(C=C, penalty='l1', random_state=self.seed)
        clf_l1.fit(X, y)
        self.L1_weights = clf_l1.coef_[0]

        self.feature_scores = self.L1_weights

        class0_features = np.nonzero(self.L1_weights < 0)[0]
        class1_features = np.nonzero(self.L1_weights > 0)[0]
        class0_features_ranked = class0_features[np.argsort(self.L1_weights[class0_features])]
        class1_features_ranked = class1_features[np.argsort(self.L1_weights[class1_features])[::-1]]
        feature_rank = (class0_features_ranked, class1_features_ranked)

        return feature_rank

    def rank_by_L1_weights(self, C, X, y):
        clf_l1 = linear_model.LogisticRegression(C=C, penalty='l1', random_state=self.seed)
        clf_l1.fit(X, y)
        self.L1_weights = clf_l1.coef_[0]
        self.feature_count = self.count_features(X, y)

        self.feature_scores = self.L1_weights

        feature_rank = np.argsort(np.absolute(self.L1_weights))[::-1]

        return self.classify_features(feature_rank)

    def classify_features(self, feature_rank):
        return self.classify_features_through_expectation(feature_rank)

    def classify_features_through_counts(self, feature_rank):
        class0_features_rank = list()
        class1_features_rank = list()

        for f in feature_rank:
            if self.feature_count[f, 1, 0] > self.feature_count[f, 1, 1]:
                class0_features_rank.append(f)
            elif self.feature_count[f, 1, 0] < self.feature_count[f, 1, 1]:
                class1_features_rank.append(f)
            # if positive and negative counts are tied, the feature is deemed
            # neither positive nor negative

        return (class0_features_rank, class1_features_rank)

    def classify_features_through_expectation(self, feature_rank):
        class0_features_rank = list()
        class1_features_rank = list()

        for f in feature_rank:

            total_count = self.feature_count[f, 1, 0] + self.feature_count[f, 1, 1]
            expected_c1_count = total_count * self.class1_prob

            if expected_c1_count < self.feature_count[f, 1, 1]:  # more class 1 than expected
                class1_features_rank.append(f)
            elif expected_c1_count > self.feature_count[f, 1, 1]:  # fewer class 1 than expected
                class0_features_rank.append(f)
            # if exactly as expected, the feature is deemed neither positive nor negative

        return (class0_features_rank, class1_features_rank)

    def count_features(self, X, y):
        X_csc = X.tocsc()
        feature_count = np.zeros(shape=(self.num_features, 2, 2))

        for f in range(self.num_features):
            feature = X_csc.getcol(f)
            nonzero_fi = feature.indices
            y_1 = np.sum(y)
            y_0 = len(y) - y_1
            feature_count[f, 1, 1] = np.sum(y[nonzero_fi])
            feature_count[f, 1, 0] = len(nonzero_fi) - feature_count[f, 1, 1]
            feature_count[f, 0, 0] = y_0 - feature_count[f, 1, 0]
            feature_count[f, 0, 1] = y_1 - feature_count[f, 1, 1]
            feature_count[f] += self.smoothing

        return feature_count

    def most_informative_feature(self, X, label):
        try:
            f = self.top_n_features(X, label, 1)[0]
        except IndexError:
            f = None
        return f

    def any_informative_feature(self, X, label):

        features = X.indices

        class_feats = self.rg.permutation(self.feature_rank[int(label)])

        for f in class_feats:
            if f in features:
                return f

        return None

    def top_n_features(self, X, label, n):
        features = X.indices

        top_features = list()
        for f in self.feature_rank[int(label)]:
            if f in features:
                top_features.append(f)
            if len(top_features) == n:
                break

        return top_features

    def top_n_class0_features(self, X, n):
        return self.top_n_features(X, 0, n)

    def top_n_class1_features(self, X, n):
        return self.top_n_features(X, 1, n)