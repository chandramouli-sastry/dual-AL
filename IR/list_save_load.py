import json
from Document import Document

def save_document_list(document_list, directory, name):
    l=[]
    for document in document_list:
        document.words = list(document.words)
        l.append(json.dumps(document.__dict__,ensure_ascii=False,encoding="latin-1"))

    with open(directory + "/" + name, "w") as f:
        json.dump(l, f)


def load_document_list(directory, name):
    with open(directory + "/" + name) as f:
        document_strings = json.load(f)
    document_list = []
    for document_string in document_strings:
        temp_doc = Document(None,None,None,None,None)
        temp_doc.__dict__ = json.loads(document_string)
        document_list.append(temp_doc)
    return document_list