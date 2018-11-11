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
from sklearn.linear_model import LogisticRegression

from CNN_Factory_DB import CNN_Factory
from RNN_Attention_Factory_DB import RNN_Attention_Factory
from list_save_load import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if os.getcwd().endswith("1shot"):
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

def train_w2v(documents, name):
    train_set = []
    for doc in documents:
        train_set.append(doc.words)
    model = Word2Vec(train_set, size=300, window=5, min_count=5, workers=8, iter=5)
    model.save(name)

def yahoo():
    dir = "yahoo"
    target_file = "documents.json"
    if target_file in os.listdir(dir):
        documents = load_document_list(dir, target_file)
    else:
        with open(dir + "/train.csv", "r") as csvfile:
            documents = []
            content = [[x[0]," ".join(x[1:])] for x in csv.reader(csvfile, delimiter=',')]
            content = random.sample(content,100000)
            for id in range(len(content)):
                class_, text = content[id]
                text = text.decode('string_escape')
                doc = Document(text=text, label="sm" if class_ == '2' else "ci",
                               correct_class=int(class_)-1, id=id)
                doc.initialise()
                documents.append(doc)
                if id % 10000==0:
                    print(id)
            print("Done. Saving...")
            documents = [doc for doc in documents if len(doc.words) > 0]
            dump_w2v(documents,"w2vyahoo.json")
            save_document_list(documents, dir, target_file)
    return documents

def yahoo_ann():
    dir = "yahoo"
    target_file = "annotated_docs.json"
    if target_file in os.listdir(dir):
        documents = load_document_list(dir, target_file)
    else:
        with open(dir + "/annotated_docs.csv", "r") as csvfile:
            documents = []
            content = [(x[0],x[1]) for x in csv.reader(csvfile, delimiter=',') if len(x)>0]
            for id in range(len(content)):
                text, class_ = content[id]
                # text = text.decode('string_escape')
                doc = Document(text=text, label=class_, correct_class=int(class_), id=id)
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
                if id % 10000 == 0:
                    print(id)
            print("Done. Saving...")
            documents = [doc for doc in documents if len(doc.words) > 0]

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

    def execute_1shot(config, boot, test_docs):
        num_classes = len(set([doc.class_ for doc in test_docs + boot]))
        print(config["description"]+str(num_classes))

        mean = np.mean([model[w] for w in model.vocab.keys()], axis=0)
        for doc in boot+test_docs:
            doc.vectors = [(model[w] if w in model else mean) for w in doc.words]

        config["args"]["num_classes"] = num_classes

        # Fetch the class object from the factory
        CLF_class = config["factory"]().get_class(**config["args"])

        for i in range(0, 10, 1):
            clf = CLF_class(model.vector_size)
            clf.train(boot)
            clf.run(test_docs)
            acc = sum([doc.class_==doc.pred_class for doc in test_docs])/(len(test_docs))
            print(acc)
            config["accuracies"].append(acc)
            with open(config["filename"] + str(i) + ".pkl", "w") as f:
                pickle.dump(config, f)
        print(config["accuracies"])
        print(np.mean(config["accuracies"]))

    def execute_1shotSharma(config, boot, test_docs):
        w2i = {}
        words = set()
        idx = 0
        for doc in boot + test_docs:
            words |= set(doc.words)

        for word in words:
            w2i[word] = idx
            idx += 1

        def vectorize(docs, annotate=False):
            data = []
            indices = []
            indptr = [0]
            targets = []
            for doc in docs:
                word_counts = {}
                imp_words = set()
                for word, term in zip(doc.words, doc.user_terms):
                    if word not in w2i:
                        continue
                    word_counts[word] = word_counts.get(word, 0) + 1
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

                indptr.append(len(indices))
                targets.append(doc.class_)

            return csr_matrix((data, indices, indptr), shape=[len(docs), len(w2i)], dtype=float), np.array(targets)

        train_vec, train_tar = vectorize(boot, annotate=False)
        test_vec, test_tar = vectorize(test_docs, annotate=False)

        clf = LogisticRegression()
        clf.fit(train_vec, train_tar)

        preds = clf.predict(test_vec)
        acc = np.sum(preds == test_tar) / test_vec.shape[0]

        print(acc)

    cnn_config = {
            "description": "CNN with user annotations; att_max = 0.8; num_classes = ",
            "factory": CNN_Factory,
            "args": {
                "use_attribution": True,
                "att_max_value": 0.8,
                "num_classes": None
            },
            "accuracies": [],
            "aucs": [],
            "filename": "CNN_prior"
        }

    rnn_config = {
            "description": "RNN with user annotations; att_max = 0.8; num_classes = ",
            "factory": RNN_Attention_Factory,
            "num_train": 100,
            "args": {
                "use_attribution": True,
                "att_max": 0.8,
                "num_classes": None
            },
            "accuracies": [],
            "aucs": [],
            "filename": "RNN_prior"
        }

    try:
        "python -m 1shot.driver [cnn|rnn|sharma]"
        import sys
        if sys.argv[1] == "cnn":
            config = cnn_config
        elif sys.argv[1] == "rnn":
            config = rnn_config
        elif sys.argv[1] == "sharma":
            config = None
            execute_1shot = execute_1shotSharma
    except Exception as e:
        config = cnn_config

    test_docs = yahoo()
    boot = yahoo_ann()
    for doc in test_docs:
        doc.class_ -=1
    for doc in boot:
        print(doc.text,doc.class_)

    dict_ = json.load(open("w2vyahoo.json", "r"), encoding="latin-1")
    model = W2VModel(dict_)
    execute_1shot(config, boot, test_docs)