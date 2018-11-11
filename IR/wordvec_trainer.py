import nltk
from gensim.models import Word2Vec#TaggedDocument, Doc2Vec
import sys
from nltk.corpus import stopwords
import re
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def refine(phrase):
    # p = p.replace("-lsb-"," ").replace("-rsb-"," ").replace("-lrb-"," ").replace("-rrb-"," ")
    p = re.sub(r"<.*?>"," ",phrase)
    p = re.sub(r"[^a-zA-Z. ]", " ", p)
    p = re.sub(r"\b[a-z]{1,2}\b", " ",p)
    p = p.lower()
    p = re.sub(r"[ ][ ]+"," ",p)
    return p

def get_tit_abs(filename):
    with open(filename, "r") as csvfile:
        content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        header = content[0]
        content = content[1:]
        DOCUMENT_TITLE = header.index(fields[0])
        CONTENT = header.index(fields[1])
        return [(doc[DOCUMENT_TITLE],doc[CONTENT]) for doc in content]

def wordvec_fit(list_all):
    train_set = []
    id = 0
    stop_w = set(stopwords.words("english"))
    for tit,doc in list_all:
        tot=refine((tit+" "+doc).decode(errors="ignore")).split()
        all_words = [word for word in tot]# if word not in tot]
        train_set.append(all_words)#,tags=[str(id)]))
        id += 1
    print len(train_set)
    sys.stdout.flush()
    model = Word2Vec(train_set,size=100, window=5, min_count=5, workers=8,iter=5)
    model.save("wv.gensim")

from codecs import open
import csv
list_all = list(csv.reader(open("data.csv",errors="ignore"),delimiter=","))
print "Hall-start",len(list_all)
list_all += get_tit_abs("hall.csv")
print "Kit-start",len(list_all)
list_all += get_tit_abs("kit.csv")
print "Wah-start",len(list_all)
list_all += get_tit_abs("wah.csv")
print "Rad-start",len(list_all)
list_all += get_tit_abs("rad.csv")
print "Total - ",len(list_all)
wordvec_fit(list_all)
