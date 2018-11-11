from __future__ import division,print_function

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
import copy

class ActiveLearningDocument:
    def compute_bm_25(self):
        b = 0.75
        k1 = 1.5

        content = [doc.title + " " + doc.text for doc in self.documents]
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False, smooth_idf=False,
                                  sublinear_tf=False, decode_error="ignore")
        tf = tfidfer.fit_transform(content)

        d_avg = np.mean(np.sum(tf, axis=1))
        score = {}
        for word in self.query:
            score[word] = []
            try:
                id = tfidfer.vocabulary_[word]
            except:
                score[word] = [0] * len(content)
                continue
            df = sum([1 for wc in tf[:, id] if wc > 0])
            idf = np.log((len(content) - df + 0.5) / (df + 0.5))
            for i in range(len(content)):
                score[word].append(
                    idf * tf[i, id] / (tf[i, id] + k1 * ((1 - b) + b * np.sum(tf[0], axis=1)[0, 0] / d_avg)))

        bm = np.sum(list(score.values()), axis=0).tolist()
        for score, doc in zip(bm, self.documents):
            doc.bm = score

        docs = sorted(self.documents, key=lambda x: x.bm, reverse=True)
        relevant_N = sum((1 for doc in docs[:self.num_relevant] if doc.class_))
        print("Prec@N with BM", relevant_N / self.num_relevant)

    def __init__(self, model, train_documents, test_documents, CLF_class, seed, query, num_train=500, train_full=False):
        self.model = model
        self.query = query
        self.seed = seed
        self.documents = train_documents
        self.test_documents = test_documents
        self.num_train = num_train
        self.num_relevant = len([doc for doc in self.documents if doc.class_==1])

        mean = np.mean([model[w] for w in model.vocab.keys()],axis=0)
        for doc in self.documents:
            doc.vectors = [(self.model[w] if w in self.model else mean) for w in doc.words]

        for doc in self.test_documents:
            doc.vectors = [(self.model[w] if w in self.model else mean) for w in doc.words]

        self.clf = CLF_class(model.vector_size)

        self.unlabeled_documents = copy.copy(train_documents)

        self.data = []
        self.labels = []

        self.relevant_documents = []
        self.irrelevant_documents = []

        self.num_sample = 10
        self.compute_bm_25()

    @property
    def yes(self):
        return len(self.relevant_documents)

    @property
    def no(self):
        return len(self.irrelevant_documents)

    def train(self):
        num_reviewed = 0

        while self.yes == 0 or self.no == 0:
            print("Bootstrapping...")
            bootstrap_sample = self.bootstrap()
            self.auto_label(bootstrap_sample)
            num_reviewed += len(bootstrap_sample)

        print("Bootstrapping done")
        iter = 0
        num_relevant = []

        while num_reviewed <= self.num_train:
            pos = self.relevant_documents
            neg = self.irrelevant_documents

            num_relevant.append(len(pos))

            print("Pos : ", len(pos))
            print("Neg : ", len(neg))
            print("Num reviewed", num_reviewed)
            print("=====================================")

            if len(pos + neg) == self.num_train:
                break

            self.clf.train(docs=pos+neg)
            self.clf.run(docs=self.documents)

            samples = self.get_certain()
            iter += 1
            self.auto_label(samples)
            num_reviewed += len(samples)

        return num_relevant

    def get_certain(self):
        docs = sorted(self.unlabeled_documents, key=lambda x: x.parameters["rel"], reverse=True)
        return docs[:self.num_sample]

    def bootstrap(self):
        return sorted(self.documents,key=lambda x:x.bm,reverse=True)[:10]

    def auto_label(self, list_documents):
        for doc in list_documents:
            if doc.class_:
                self.relevant_documents.append(doc)
            else:
                self.irrelevant_documents.append(doc)
            self.unlabeled_documents.remove(doc)