from __future__ import division, print_function

import logging
import os
import pickle
import random
import re

import numpy as np
import csv

from gensim.models import Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score



from CNN_Factory import CNN_Factory
from RNN_Attention_Factory import RNN_Attention_Factory
# from .. import list_save_load
from feature_expert import feature_expert
from list_save_load import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sw=set(stopwords.words("english"))

if os.getcwd().endswith("featExpert_AL"):
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

    model = Word2Vec(size=300, window=5, min_count=5, workers=8)
    model.build_vocab(train_set)
    # model.intersect_word2vec_format("../GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)
    model.train(train_set,total_examples=len(train_set),epochs=15)

    if name is not None:
        model.save(name)
    return model

def extract_features(texts,classes):
    vect = CountVectorizer(analyzer=u'word', binary= True, decode_error=u'ignore',
        lowercase=True, max_df=1.0, max_features=None, min_df=5,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)

    X = vect.fit_transform(texts)
    idx =np.where(X.sum(axis=1) > 0)[0]
    X = X[idx]
    y = np.array(classes)[idx]

    fe = feature_expert(X,y,metric='chi2',pick_only_top=True)

    i2w = {vect.vocabulary_[w]:w for w in vect.vocabulary_}
    pos_words = [i2w[i] for i in fe.feature_rank[0]]
    neg_words = [i2w[i] for i in fe.feature_rank[1]]

    return pos_words, neg_words

def initialize(data,targets,pos_words,neg_words):
    documents = []
    for datum,target in zip(data,targets):
        doc = Document(text=datum,label="neg" if target==0 else "pos",correct_class=target,id=0)
        doc.initialise()
        doc.words = [word for word in doc.words if word not in sw]
        rel_words = pos_words if target==0 else neg_words
        possible_words = list(set(doc.words)&set(rel_words))
        word = random.choice(possible_words) if len(possible_words)!=0 else None
        doc.user_terms = [(w==word)*1 for w in doc.words]
        documents.append(doc)
    return [doc for doc in documents if len(doc.words)>0]


def WvsH():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers'),
                                          categories=["comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware"])

    pos_words, neg_words = extract_features(newsgroups_train.data, newsgroups_train.target)

    train_docs = initialize(newsgroups_train.data, newsgroups_train.target,pos_words,neg_words)

    test_docs = train_docs
    train_docs = random.sample([doc for doc in train_docs if doc.class_==1],5) + random.sample([doc for doc in train_docs if doc.class_==0],5)

    all_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    docs = initialize(all_train.data, all_train.target, pos_words,neg_words)
    model = train_w2v(docs, None)
    return train_docs,test_docs, model.wv

def nova():
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers'))

    texts = []
    targets = []

    for i in range(len(data.data)):
        texts.append(data.data[i])
        target = 1*("talk." in data.target_names[data.target[i]] or ".religion." in data.target_names[data.target[i]] or "atheism" in data.target_names[data.target[i]])
        targets.append(target)

    pos_words, neg_words = extract_features(texts,targets)

    train_docs = initialize(texts, targets, pos_words, neg_words)
    model = train_w2v(train_docs, None)

    return train_docs, train_docs, model.wv

def sraa():
    import re
    def remove_headers(text):
        return text
        return re.sub("((.*?):.*\n)","",text)

    def read_all(path):
        files = os.listdir(path)
        texts = []
        for f in files:
            texts.append(remove_headers(open(os.path.join(path,f)).read()))
        return texts

    autos = read_all("sraa/realauto")
    autos += read_all("sraa/simauto")
    aviations = read_all("sraa/simaviation")
    aviations += read_all("sraa/realaviation")

    texts = autos+aviations
    targets = [0]*len(autos)+[1]*len(aviations)

    pos_words, neg_words = extract_features(texts,targets)

    train_docs = initialize(texts, targets, pos_words, neg_words)
    model = train_w2v(train_docs, None)

    return train_docs, train_docs, model.wv

if __name__ == "__main__":
    class W2VModel:
        def __init__(self, dict_):
            self.vocab = {w: np.array(dict_[w]) for w in dict_}
            self.vector_size = len(list(dict_.values())[0])

        def __getitem__(self, item):
            return self.vocab[item]

        def __contains__(self, item):
            return item in self.vocab

    def execute(config, boot, test_docs):

        print(config["description"])

        mean = np.mean([model[w] for w in model.vocab.keys()], axis=0)
        for doc in boot+test_docs:
            doc.vectors = [(model[w] if w in model else mean) for w in doc.words]

        # Fetch the class object from the factory
        CLF_class = config["factory"]().get_class(**config["args"])

        for i in range(0, 50, 1):
            boot = random.sample([doc for doc in test_docs if doc.class_==1],5) + random.sample([doc for doc in test_docs if doc.class_==0],5)
            clf = CLF_class(model.vector_size)
            clf.train(boot)
            clf.run(test_docs)
            correct = 0
            true = []
            score = []
            pred = []
            for doc in test_docs:
                true.append(doc.class_)
                score.append(doc.parameters["rel"])
                pred.append(doc.pred_class)
                if (doc.class_ == doc.pred_class):
                    correct += 1
            acc = correct / (len(test_docs))
            try:
                auc = roc_auc_score(true, score)
            except Exception as e:
                auc = 0
            config["accuracies"].append(acc)
            config["aucs"].append(auc)
            print(acc)
            print(auc)
            with open(config["filename"] + str(i) + "_".join(map(str,classes)) + ".pkl", "w") as f:
                pickle.dump(config, f)
        print(config["accuracies"])
        print(np.mean(config["accuracies"]))

        # print(config["aucs"])


    cnn_configs = [
        {
            "description": "CNN without user annotations",
            "factory": CNN_Factory,
            "args": {
                "use_attribution": False
            },
            "accuracies": [],
            "aucs": [],
            "filename": "CNN_normal"
        },
        {
            "description": "CNN with user annotations; att_max = 0.8",
            "factory": CNN_Factory,
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
            "description": "RNN without user annotations",
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
            "description": "RNN with user annotations; att_max = 0.8;",
            "factory": RNN_Attention_Factory,
            "num_train": 100,
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
        "python -m featExpert_AL.driver [cnn|rnn] [0|1] [nova|sraa|WvsH]"
        import sys
        if sys.argv[1] == "cnn":
            config = cnn_configs[int(sys.argv[2])]
        else:
            config = rnn_configs[int(sys.argv[2])]
        func = eval(sys.argv[3]) # func should be one of ["nova","sraa","WvsH"]
    except Exception as e:
        # CNN - 0 3
        # RNN - 0 1 2 3
        config = rnn_configs[1]
        classes = range(10)
        func = WvsH

    boot, test_docs, model = func()
    execute(config, boot, test_docs)