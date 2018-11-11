from __future__ import division, print_function
import os
import pickle

import numpy as np
import csv
import logging

from gensim.models import KeyedVectors

from CNN_Factory import CNN_Factory
from RNN_Attention_Factory import RNN_Attention_Factory
from list_save_load import *
from Active_Learning import ActiveLearningDocument

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if os.getcwd().endswith("IR"):
    os.chdir("..")

def ir(dir):
    target_file = "documents.json"
    files = os.listdir(dir)
    if target_file in files:
        return load_document_list(dir, target_file)
    else:
        ut = user_terms[dir]
        with open(dir + "/data.csv", "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        header = content[0]
        content = content[1:]
        DOCUMENT_TITLE = header.index(fields[0])
        CONTENT = header.index(fields[1])
        LABEL = header.index("label")
        documents = [Document(text= doc[CONTENT], label=doc[LABEL],correct_class= 1*(doc[LABEL]=="yes"), id=ind, title=doc[DOCUMENT_TITLE]) for ind,doc in enumerate(content)]
        count = 0
        perc = 1
        for doc in documents:
            doc.initialise()
            all_terms = sorted(list(set(doc.words)))
            w2i = {w:i for i,w in enumerate(all_terms)}
            doc.gb = [[(i==w)*1 for i in doc.words] for w in all_terms]
            doc.user_terms = [0]*len(w2i)
            doc.ut2 = [0]*len(doc.words)
            if doc.label== "yes":
                for i,w in enumerate(doc.words):
                    if w in ut:
                        doc.ut2[i] = 1
            if doc.label:
                for w in ut:
                    if w in w2i:
                        doc.user_terms[w2i[w]] = 1
            doc.w2i = w2i
            if (count/len(documents))*100>perc:
                print(count/len(documents)*100)
                perc += 1
        save_document_list(documents,dir,"documents.json")
        return documents

if __name__ == "__main__":
    cnn_configs = [
        {
            "description": "CNN without user annotations",  # 000
            "factory": CNN_Factory,
            "num_train": 50,
            "args": {
                "use_attention": False
            },
            "num_relevant": [],
            "filename": "CNN_normal"
        },
        {
            "description": "CNN with user annotations with softmax and occlusion; limit = 0.5",  # 100
            "factory": CNN_Factory,
            "num_train": 50,
            "args": {
                "use_attention": True,
                "att_max_value": 0.4
            },
            "num_relevant": [],
            "filename": "CNN_prior_5"
        }
    ]

    rnn_configs = [
        {
            "description": "RNN without user annotations",  # 0000
            "factory": RNN_Attention_Factory,
            "num_train": 50,
            "args": {
                "use_attention": False
            },
            "num_relevant": [],
            "filename": "RNN_normal"
        },
        {
            "description": "RNN with user annotations with softmax with att_max = 0.8",  # 1000
            "factory": RNN_Attention_Factory,
            "num_train": 50,
            "args": {
                "use_attention": True,
                "att_max": 0.4
            },
            "num_relevant": [],
            "filename": "RNN_user_perc_8_rep"
        }
    ]
    try:
        "python -m IR.driver [cnn|rnn] [0|1] [Hall|Kitchenham|Wahano|Radjenovic]"
        import sys
        dir = sys.argv[3]
        if sys.argv[1] == "cnn":
            config = cnn_configs[int(sys.argv[2])]
        else:
            config = rnn_configs[int(sys.argv[2])]
    except Exception as e:
        dir = "Kitchenham"
        config= cnn_configs[1]


    class Model:
        def __init__(self, dict_):
            self.vocab = {w: np.array(dict_[w]) for w in dict_}
            self.vector_size = len(list(dict_.values())[0])

        def __getitem__(self, item):
            return self.vocab[item]

        def __contains__(self, item):
            return item in self.vocab


    query = {
        "Hall": "defect prediction",
        "Kitchenham": "systematic literature review",
        "Wahano": "defect prediction",
        "Radjenovic": "defect prediction metrics"
    }

    user_terms = {
        "Hall": ["software","defect","prediction","performance"],
        "Wahano": ["software", "defect","prediction", "dataset", "framework"],
        "Kitchenham": ["systematic","literature","review"],
        "Radjenovic": ["metrics","software","defect","prediction"]
    }

    docs = ir(dir)
    dict_ = json.load(open("wv.json", "r"), encoding="latin-1")
    print(len(docs))

    model = KeyedVectors.load("wv.gensim").wv

    def execute_config(config):
        print(config["description"])
        print(dir)
        CLF_class = config["factory"]().get_class(**config["args"])
        for j in range(10):
            train = docs
            al = ActiveLearningDocument(model=model, train_documents=train, test_documents=train, CLF_class=CLF_class, seed=j,
                                        num_train=config["num_train"],query=query[dir].split())
            num_relevant = al.train()
            print(num_relevant)
            config["num_relevant"].append(num_relevant)
            with open(config["filename"] + str(dir)+".pkl", "w") as f:
                pickle.dump(config, f)
        print(config["num_relevant"])

    execute_config(config)
