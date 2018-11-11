from __future__ import division, print_function

import logging
import os
import pickle
import random
import re

import numpy as np
import csv

from gensim.models import Word2Vec, KeyedVectors
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from CNN_Factory import CNN_Factory
from RNN_Attention_Factory import RNN_Attention_Factory
from Active_Learning import ActiveLearning
from list_save_load import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if os.getcwd().endswith("IR"):
    os.chdir("..")

def dump_w2v(documents, name):
    import json
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    print("loaded")
    word_vec = {}
    for doc in documents:
        for word in doc.words:
            if word in model:
                word_vec[word] = model[word].tolist()
    f = open(name, "w")
    json.dump(word_vec, f)
    f.close()
    return word_vec


def train_w2v(documents, name):
    train_set = []
    for doc in documents:
        train_set.append(doc.words)
    model = Word2Vec(train_set, size=300, window=5, min_count=5, workers=8, iter=5)
    model.save(name)


def pang_imdb_train():
    """
    :return: Returns list of train documents
    """
    dir = "pangimdb"
    target_file = "documents.json"
    files = os.listdir(dir)
    if target_file in files:
        return load_document_list(dir, target_file)
    else:
        documents = []
        num_classes = 0
        for x in os.listdir(dir + "/withRats_neg"):
            text = open(os.path.join(dir, "withRats_neg", x)).read()
            doc = Document(text=text, label="neg", correct_class=0, id=x)
            doc.initialise()
            assign = False
            words = []
            user_terms = []
            for idx, word in enumerate(doc.words):
                if word == "neg":
                    assign = True
                    continue
                if word == "/neg":
                    assign = False
                    continue
                words.append(word)
                if assign:
                    user_terms.append(1)
                else:
                    user_terms.append(0)
            doc.words = words
            doc.user_terms = user_terms
            documents.append(doc)
        for x in os.listdir(dir + "/withRats_pos"):
            text = open(os.path.join(dir, "withRats_pos", x)).read()
            doc = Document(text=text, label="pos", correct_class=1, id=x)
            doc.initialise()
            assign = False
            words = []
            user_terms = []
            for idx, word in enumerate(doc.words):
                if word == "pos":
                    assign = True
                    continue
                if word == "/pos":
                    assign = False
                    continue
                words.append(word)
                if assign:
                    user_terms.append(1)
                else:
                    user_terms.append(0)
            doc.words = words
            doc.user_terms = user_terms
            documents.append(doc)

        save_document_list(documents, dir, "documents.json")
        dump_w2v(documents,"w2vimdb.json")
        return documents


def pang_imdb_test():
    """
    :return: Returns list of test documents
    """
    dir = "pangimdb"
    target_file = "test_documents.json"
    files = os.listdir(dir)
    if target_file in files:
        return load_document_list(dir, target_file)
    else:
        documents = []
        num_classes = 0
        for x in os.listdir(dir + "/noRats_neg"):
            if not (900 <= int(x[4:7]) <= 999):
                continue
            text = open(os.path.join(dir, "noRats_neg", x)).read()
            doc = Document(text=text, label="neg", correct_class=0, id=x)
            doc.initialise()
            documents.append(doc)
        for x in os.listdir(dir + "/noRats_pos"):
            if not (900 <= int(x[4:7]) <= 999):
                continue
            text = open(os.path.join(dir, "noRats_pos", x)).read()
            doc = Document(text=text, label="pos", correct_class=1, id=x)
            doc.initialise()
            documents.append(doc)
        save_document_list(documents, dir, "test_documents.json")
        return documents


def acl_imdb_train():
    """
        :return: Returns list of train documents
    """
    dir = "aclimdb"
    target_file = "documents.json"
    files = os.listdir(dir)
    if target_file in files:
        return load_document_list(dir, target_file)
    else:
        documents = []
        num_classes = 0
        for x in os.listdir(dir + "/train/neg"):
            text = open(os.path.join(dir, "train/neg", x)).read()
            doc = Document(text=text, label="neg", correct_class=0, id=x)
            doc.initialise()
            documents.append(doc)
        for x in os.listdir(dir + "/train/pos"):
            text = open(os.path.join(dir, "train/pos", x)).read()
            doc = Document(text=text, label="pos", correct_class=1, id=x)
            doc.initialise()
            documents.append(doc)
        save_document_list(documents, dir, "documents.json")
        dump_w2v(documents,"w2vacl.json")

        return documents


def acl_imdb_test():
    """
        :return: Returns list of train documents
    """
    dir = "aclimdb"
    target_file = "test_documents.json"
    files = os.listdir(dir)
    if target_file in files:
        return load_document_list(dir, target_file)
    else:
        documents = []
        num_classes = 0
        for x in os.listdir(dir + "/test/neg"):
            text = open(os.path.join(dir, "test/neg", x)).read()
            doc = Document(text=text, label="neg", correct_class=0, id=x)
            doc.initialise()
            documents.append(doc)
        for x in os.listdir(dir + "/test/pos"):
            text = open(os.path.join(dir, "test/pos", x)).read()
            doc = Document(text=text, label="pos", correct_class=1, id=x)
            doc.initialise()
            documents.append(doc)
        save_document_list(documents, dir, "test_documents.json")
        return documents

if __name__ == "__main__":
    class W2VModel:
        def __init__(self, dict_):
            self.vocab = {w: np.array(dict_[w]) for w in dict_}
            self.vector_size = len(list(dict_.values())[0])

        def __getitem__(self, item):
            return self.vocab[item]

        def __contains__(self, item):
            return item in self.vocab


    def stratified_kfold(docs, s):
        """
        Stratified k-fold
        :param docs: List of training documents to partition
        :param s: The Random Seed
        :return: Stratified k-fold (k=10)
        """
        r = random.Random(s)
        pos = [doc for doc in docs if doc.class_ == 1]
        neg = [doc for doc in docs if doc.class_ == 0]
        r.shuffle(pos)
        r.shuffle(neg)
        l = int(len(pos) / 10)
        return [pos[i * l:(i + 1) * l] + neg[i * l:(i + 1) * l] for i in range(10)]


    def execute_IMDB(config, seeds, train_docs, test_docs):
        print(config["description"])
        print(seeds)
        # Fetch the class object from the factory
        CLF_class = config["factory"]().get_class(**config["args"])
        for j in seeds:
            folds = stratified_kfold(train_docs, j)
            for i in range(0, 10, 1):
                test = folds[i]
                train = sum(folds[:i] + folds[i + 1:], [])
                al = ActiveLearning(model=model,
                                    train_documents=train,
                                    # Use both unannotated docs and i-th fold as held-out test dataset
                                    test_documents=test_docs + test,
                                    CLF_class=CLF_class,
                                    seed=i + 1,
                                    num_train=config["num_train"])
                accs, aucs = al.train()
                config["accuracies"].append(accs)
                config["aucs"].append(aucs)
                with open(config["filename"] + str(j) + "IMDB.pkl", "w") as f:
                    pickle.dump(config, f)
        print(config["accuracies"])
        print(config["aucs"])


    def execute_ACL(config, train_docs, test_docs, train_docs_pang):
        print(config["description"])

        # Fetch the class object from the factory
        CLF_class = config["factory"]().get_class(**config["args"])
        for i in range(0, 5, 1):
            boot = random.sample([doc for doc in train_docs_pang if doc.class_ == 0], 5) + random.sample(
                [doc for doc in train_docs_pang if doc.class_ == 1], 5)
            al = ActiveLearning(model=model,
                                train_documents=train_docs + boot,
                                # Use both unannotated docs and i-th fold as held-out test dataset
                                test_documents=test_docs,
                                CLF_class=CLF_class,
                                seed=i + 1,
                                num_train=config["num_train"],
                                boot=boot)
            accs, aucs = al.train()
            config["accuracies"].append(accs)
            config["aucs"].append(aucs)
            with open(config["filename"] + str(i) + "ACL.pkl", "w") as f:
                pickle.dump(config, f)
        print(config["accuracies"])
        print(config["aucs"])

    def execute_ACLSharma(config, train_docs, test_docs, train_docs_pang):
        print(config["description"])

        w2i = {}
        words = set()
        idx = 0
        for doc in train_docs_pang+train_docs:
            words |= set(doc.words)

        for word in words:
            w2i[word] = idx
            idx += 1

        def vectorize(docs,annotate=False):
            data = []
            indices = []
            indptr = [0]
            targets = []
            for doc in docs:
                word_counts = {}
                imp_words = set()
                for word,term in zip(doc.words,doc.user_terms):
                    if word not in w2i:
                        continue
                    word_counts[word] = word_counts.get(word,0) + 1
                    if term == 1:
                        imp_words.add(word)
                if annotate:
                    for word in set(doc.words):
                        if word not in w2i:
                            continue
                        if word not in imp_words:
                            word_counts[word] *= 0.01

                for word in word_counts:
                    data.append(word_counts[word])
                    indices.append(w2i[word])

                # data.append(datum)
                # indices.append(idx)
                indptr.append(len(indices))
                targets.append(doc.class_)

            return csr_matrix((data,indices,indptr),shape=[len(docs),len(w2i)],dtype=float), np.array(targets)

        train_vec, train_tar = vectorize(train_docs_pang,annotate=False)
        test_vec, test_tar = vectorize(train_docs,annotate=False)

        # Fetch the class object from the factory
        CLF_class = config["factory"]().get_class(**config["args"])
        for i in range(0, 5, 1):
            boot_indices = random.sample([idx for idx,doc in enumerate(train_docs_pang) if doc.class_ == 0], 5) + random.sample(
                [idx for idx,doc in enumerate(train_docs_pang) if doc.class_ == 1], 5)

            boot = train_vec[boot_indices]
            boot_targets = train_tar[boot_indices]

            clf = LogisticRegression()
            clf.fit(boot,boot_targets)
            preds = clf.predict(test_vec)
            probs = clf.predict_proba(test_vec)
            acc = np.sum(preds==test_tar)/test_vec.shape[0]
            auc = roc_auc_score(test_tar,probs[:,1])
            print(acc)
            print(auc)
            config["accuracies"].append(acc)
            config["aucs"].append(auc)
            with open(config["filename"] + str(i) + "ACL.pkl", "w") as f:
                pickle.dump(config, f)
        print(config["accuracies"])
        print(config["aucs"])

    cnn_configs = [
        {
            "description": "CNN without user annotations; budget = 500",
            "factory": CNN_Factory,
            "args": {
                "use_attribution": False
            },
            "num_train": 500,
            "accuracies": [],
            "aucs": [],
            "filename": "CNN_normal"
        },
        {
            "description": "CNN with user annotations; att_max = 0.8; budget = 250",
            "factory": CNN_Factory,
            "num_train": 250,
            "args": {
                "use_attribution": True,
                "att_max_value": 0.8
            },
            "accuracies": [],
            "aucs": [],
            "filename": "CNN_prior"
        },
        {
            "description": "CNN without user annotations; budget = 50",
            "factory": CNN_Factory,
            "args": {
                "use_attribution": False
            },
            "num_train": 50,
            "accuracies": [],
            "aucs": [],
            "filename": "CNN_normal"
        },
        {
            "description": "CNN with user annotations; att_max = 0.8; budget = 40",
            "factory": CNN_Factory,
            "num_train": 40,
            "args": {
                "use_attribution": True,
                "att_max_value": 0.8
            },
            "accuracies": [],
            "aucs": [],
            "filename": "CNN_prior"
        }
    ]

    rnn_configs = [
        {
            "description": "RNN without user annotations; budget = 200",
            "factory": RNN_Attention_Factory,
            "num_train": 200,
            "args": {
                "use_attribution": False
            },
            "accuracies": [],
            "aucs": [],
            "filename": "RNN_normal"
        },
        {
            "description": "RNN with user annotations; att_max = 0.8; budget = 100",
            "factory": RNN_Attention_Factory,
            "num_train": 100,
            "args": {
                "use_attribution": True,
                "att_max": 0.8
            },
            "accuracies": [],
            "aucs": [],
            "filename": "RNN_prior"
        },
        {
            "description": "RNN without user annotations; budget = 50",
            "factory": RNN_Attention_Factory,
            "num_train": 50,
            "args": {
                "use_attribution": False
            },
            "accuracies": [],
            "aucs": [],
            "filename": "RNN_normal"
        },
        {
            "description": "RNN with user annotations; att_max = 0.8; budget = 40",
            "factory": RNN_Attention_Factory,
            "num_train": 40,
            "args": {
                "use_attribution": True,
                "att_max": 0.8
            },
            "accuracies": [],
            "aucs": [],
            "filename": "RNN_prior"
        }
    ]

    try:
        "python -m classification.driver [pangimdb|aclimdb] [cnn|rnn] [0|1]"
        "python -m classification.driver aclimdb_sharma"
        import sys
        dir = sys.argv[1]
        if dir == "pangimdb":
            if sys.argv[2] == "cnn":
                config = cnn_configs[int(sys.argv[3])]
                seeds = range(5)
            else:
                config = rnn_configs[int(sys.argv[3])]
                seeds = range(int(sys.argv[4]), int(sys.argv[5]))
        elif dir == "aclimdb":
            if sys.argv[2] == "cnn":
                config = cnn_configs[int(sys.argv[3])]
            else:
                config = rnn_configs[int(sys.argv[3])]
        else:
            config = None
            seeds = None
    except Exception as e:
        dir = "aclimdb_sharma"
        config = rnn_configs[1]
        # config = rnn_configs[1]
        seeds = range(5)

    if dir == "pangimdb":
        train_docs = pang_imdb_train()
        test_docs = pang_imdb_test()
        try:
            dict_ = json.load(open("w2v.json", "r"), encoding="latin-1")
        except Exception as E:
            dict_ = dump_w2v(train_docs + test_docs, "w2v.json")

        model = W2VModel(dict_)
    elif dir == "aclimdb":
        train_docs_pang = pang_imdb_train()
        train_docs = acl_imdb_train()
        test_docs = acl_imdb_test()
        try:
            dict_ = json.load(open("w2vacl.json", "r"), encoding="latin-1")
        except Exception as E:
            dict_ = dump_w2v(train_docs + test_docs, "w2vacl.json")
        model = W2VModel(dict_)
        execute_ACL(config, train_docs, test_docs, train_docs_pang)
    elif dir == "aclimdb_sharma":
        train_docs_pang = pang_imdb_train()
        train_docs = acl_imdb_train()
        test_docs = acl_imdb_test()
        execute_ACLSharma(config, train_docs, test_docs, train_docs_pang)

    """
    pangimdb cnn 0
    pangimdb cnn 1
    pangimdb rnn 0 0 5
    
    aclimdb cnn 2
    aclimdb cnn 3
    aclimdb rnn 2
    aclimdb rnn 3
    """