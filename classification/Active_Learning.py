from __future__ import division,print_function

import random
import numpy as np
import math
from sklearn.metrics import roc_auc_score,f1_score
from collections import Counter
import copy

class ActiveLearning:
    def __init__(self, model, train_documents, test_documents, CLF_class, seed, num_train=500, boot=None):
        self.model = model # The Word2Vec model
        self.seed = seed # Seed for selecting Bootstrap Sample
        self.documents = train_documents # Training Document list
        self.test_documents = test_documents # Held out test data
        self.num_train = num_train # The budget
        self.boot = boot # Pre-determined bootstrap sample (if any)

        # Use the mean of the word embeddings as word vector for OOV (Out-of-vocab) words
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

        # Bootstrap Sample
        self.num_sample = 10

    @property
    def yes(self):
        return len(self.relevant_documents)

    @property
    def no(self):
        return len(self.irrelevant_documents)

    def train(self):
        num_reviewed = 0
        if self.boot is None:
            bootstrap_sample = self.bootstrap()
        else:
            bootstrap_sample = self.boot
        self.auto_label(bootstrap_sample)
        num_reviewed += len(bootstrap_sample)

        print("Bootstrapping done")
        iter = 0
        accuracies = []
        aucs = []

        while num_reviewed <= self.num_train:
            pos = self.relevant_documents
            neg = self.irrelevant_documents

            self.train_pos = pos
            self.train_neg = neg

            train_docs = self.train_pos + self.train_neg
            self.train_docs = train_docs

            self.clf.train(docs=train_docs)
            self.clf.run(docs=self.documents + self.test_documents)

            correct = 0
            true = []
            score = []
            pred = []
            for doc in self.test_documents:
                true.append(doc.class_)
                score.append(doc.parameters["rel"])
                pred.append(doc.pred_class)
                if (doc.class_==doc.pred_class):
                    correct += 1

            accuracy = correct / (len(self.test_documents))
            try:
                auc = roc_auc_score(true, score)
            except Exception as e:
                auc = 0
                print(e)
            try:
                f1 = f1_score(true, pred)
            except Exception as e:
                f1 = 0
                print(e)

            print("Accuracy", accuracy)
            print("AUC", auc)
            print("F1", f1)
            print("Num reviewed", num_reviewed)
            print("=====================================")

            accuracies.append(accuracy)
            aucs.append(auc)

            self.iter = iter
            samples = self.get_uncertain()
            iter += 1
            self.auto_label(samples)
            num_reviewed += len(samples)

        self.accuracy = accuracy
        print("Accuracy - ", accuracy)
        print("AUC - ", auc)
        return accuracies,aucs

    def get_uncertain(self):
        # Choose those documents which have the greatest prediction entropy
        for doc in self.documents:
            rel = doc.parameters["rel"]
            doc.unc = -(rel*math.log(rel+10**-5,2)+(1-rel)*math.log(1-rel+10**-5,2))
        docs = sorted(self.unlabeled_documents, key=lambda x: x.unc, reverse=True)
        return docs[:self.num_sample]

    def bootstrap(self):
        #Choose 5 random positive and 5 random negative documents
        R = random.Random(self.seed)
        pos = list(filter(lambda doc:doc.class_==0 and sum(doc.user_terms)>0 and doc.id[:3]=="neg",self.documents))
        if len(pos)==0:
            pos = list(filter(lambda doc:doc.class_==0,self.documents))
        neg = list(filter(lambda doc:doc.class_==1 and sum(doc.user_terms)>0 and doc.id[:3]=="pos",self.documents))
        if len(neg)==0:
            neg = list(filter(lambda doc:doc.class_==1,self.documents))
        return R.sample(pos,5) + R.sample(neg,5)

    def auto_label(self, list_documents):
        # Programmatically label documents.
        for doc in list_documents:
            if doc.class_:
                self.relevant_documents.append(doc)
            else:
                self.irrelevant_documents.append(doc)
            self.unlabeled_documents.remove(doc)