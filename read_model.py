import numpy as np

def read_vocab(filename):
    import codecs
    ''' Remember to open file f with the right encoding'''
    f = open(filename, "r")
    encoding = f.readline()
    f = codecs.open(filename, "r", encoding)
    vocab = []
    for line in f:
        word = line.split('\n')[0]
        vocab.append(word)
    f.close()
    return vocab


def read_docs(filename):
    f = open(filename, "r")
    docs = []
    for line in f:
        doc = line.split('\n')[0]
        docs.append(doc)
    f.close()
    return docs


def read_inverted_file(filename):
    f = open(filename, "r")
    inverted_file_terms = []
    inverted_file_postings = []
    for line in f:
        line = line.split('\n')[0]
        vocab_id_1, vocab_id_2, n = line.split(' ')
        inverted_file_terms.append((int(vocab_id_1), int(vocab_id_2)))
        postings = []
        for _ in range(int(n)):
            line = f.readline()
            line = line.split('\n')[0]
            doc_id, cnt = line.split(' ')
            postings.append((int(doc_id), int(cnt)))
        inverted_file_postings.append(postings)
    f.close()
    return inverted_file_terms, inverted_file_postings


def pickle_models(vocab, docs, inverted_file, filename):
    import pickle
    f = open(filename, "wb")
    pickle.dump((vocab, docs, inverted_file), f)


def load_models_from_pickle(filename):
    import pickle
    f = open(filename, "rb")
    return pickle.load(f)


if __name__ == "__main__":
    # Vocabulary
    #   - Read vocab.all
    vocab = read_vocab("./model/vocab.all")
    print(vocab[:10])

    # Documents
    #   - Read file-list
    docs = read_docs("./model/file-list")
    print(docs[:10])
     
    # Indexing (Done for you)
    #   - Read inverted-file (postings list)
    inverted_file_terms, inverted_file_postings = read_inverted_file("./model/inverted-file")
    print(inverted_file_terms[:10])
    print(inverted_file_postings[:10])