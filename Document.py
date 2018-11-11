import nltk
import re
pat=re.compile("[a-z]+")

class Document:
    def __init__(self,text,label,correct_class,id):
        try:
            self.text = unicode(text.lower(),errors="ignore").encode("ascii",errors="ignore") if text is not None else text
        except TypeError:
            self.text = text.lower() if text is not None else text
        self.label = label
        self.class_ = correct_class
        self.id = id

    def initialise(self):
        # Tokenize the text into words
        self.words = nltk.word_tokenize(self.text)
        self.words = [w for w in self.words]
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