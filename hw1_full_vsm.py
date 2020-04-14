import math
import copy
import time
import numpy
import read_model

'''
Every thing we have to do
Rules are marked with 'Rule:' in the comments
'''

mode = 'train'


def create_zeroed_2D_matrix(i,j):
    A = []
    column = [0] * j
    for row in range(i):
        A.append(copy.deepcopy(column))
        if row%1000==0:
            print(row)
    return A


if __name__ == "__main__":
    
    '''
    Read in the models.

    - Read Vocabulary (vocab.all) into a list
    - Read Documents (file-list) into a list
    - Read Inverted File (inverted-file) into two lists, one for 'terms' and one for 'postings'
    '''

    start_time = time.time()
    print('Reading models...', end='', flush=True)
    vocab = read_model.read_vocab("./model/vocab.all")

    docs = read_model.read_docs("./model/file-list")
    
    invf_terms, invf_postings = read_model.read_inverted_file("./model/inverted-file")
    print(f'done. ({time.time()-start_time}sec)')


    '''
    Construct the Vector Space

    - i axis is all the 'term'
    - j axis is all the 'document'
    - weight is the 'TF-IDF' (tf:term frequency, idf:inverse document frequency)
    - LSI?

    Apparently, Not Viable (list size way too big!)
    '''

    start_time = time.time()
    print(f'Constructing the Vector Space [{len(invf_terms)},{len(docs)}]...', end='', flush=True)
    VS = create_zeroed_2D_matrix(len(invf_terms), len(docs))
    
    
    
    
    print(f'done. ({time.time()-start_time}sec)')


    '''

    '''
