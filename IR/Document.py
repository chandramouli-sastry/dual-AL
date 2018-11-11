import nltk
import re
import nltk
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))

def refine(phrase):
    p = phrase.lower()
    # p = p.replace("-lsb-"," ").replace("-rsb-"," ").replace("-lrb-"," ").replace("-rrb-"," ")
    p = re.sub("[^a-zA-Z. ]", " ", p)
    p = re.sub("[ ][ ]+"," ",p)
    return p

class Document:
    def __init__(self,text,label,correct_class,id,title):
        self.text = unicode(text,errors="ignore").encode("ascii",errors="ignore") if text is not None else text
        self.label = label
        self.class_ = correct_class
        self.id = id
        self.title = unicode(title,errors="ignore").encode("ascii",errors="ignore") if title is not None else title

    def initialise(self):
        self.title_words = nltk.word_tokenize(self.title.lower())
        self.text_words = nltk.word_tokenize(self.text.lower())
        self.words = [w for w in self.title_words+self.text_words if len(w)>2 and w not in sw]
        self.user_terms = [0 for w in self.words]

    def remove_label(self):
        new_user_terms = []
        new_words = []
        for word,user_term in zip(self.words,self.user_terms):
            if not(word == "NEG" or word == "/NEG" or word == "POS" or word == "/POS"):
                new_words.append(word)
                new_user_terms.append(user_term)
        self.words = new_words
        self.user_terms = new_user_terms

    def __contains__(self, item):
        return item in self.text.lower()