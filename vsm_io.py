import numpy as np


def parse_sys_arguments(argv):
    rel_on, input_query_filename, output_ranked_filename, model_dir, ntcir_dir = None, None, None, None, None
    last = None
    for s in argv[1:]:
        if last == None:
            if s == '-r':
                rel_on = True
            else:
                last = s
        else:
            if last == '-i':
                input_query_filename = s
            elif last == '-o':
                output_ranked_filename = s
            elif last == '-m':
                model_dir = s
            elif last == '-d':
                ntcir_dir = s
            last = None
    return rel_on, input_query_filename, output_ranked_filename, model_dir, ntcir_dir


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


def read_doclen(filename):
    f = open(filename, "r")
    doclen = []
    for line in f:
        l = line.split('\n')[0]
        doclen.append(int(l))
    f.close()
    return doclen


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
    f.close()


def load_models_from_pickle(filename):
    import pickle
    f = open(filename, "rb")
    return pickle.load(f)
    f.close()


def write_rank(f, query_number, docs):
    f.write(f'{query_number[-3:]},')
    for doc in docs[:-1]:
        f.write(f'{doc.lower()} ')
    f.write(f'{docs[len(docs)-1].lower()}\n')

