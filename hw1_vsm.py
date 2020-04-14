import math
import copy
import time

import numpy

import query_processing as qp
import vsm
import read_model


'''
Rules are marked with 'Rule:' in the comments
'''

mode = 'train'
should_read_models = False


if __name__ == "__main__":
    
    '''
    Read in the models.

    - Read Vocabulary (vocab.all) into a list
    - Read Documents (file-list) into a list
    - Read Inverted File (inverted-file) into two lists, one for 'terms' and one for 'postings'
    '''

    start_time = time.time()
    print('Reading models...', end='', flush=True)
    if should_read_models:
        vocab = read_model.read_vocab("./model/vocab.all")

        docs = read_model.read_docs("./model/file-list")
    
        invf_terms, invf_postings = read_model.read_inverted_file("./model/inverted-file")
    print(f'done. ({time.time()-start_time}sec)')


    '''
    Process The Query

    - Read Query
    - For Every Query, Do...
        - Query -> Term (in the form of "inverted file index")
            * Title -> Unigram/Bigram
            * Concepts -> Unigram/Bigram
            * Stemming? (Rocchio?)
        
    '''
    
    
    queries = qp.read_queries(f"./queries/query-{mode}.xml")

    for query in queries:
        
        start_time = time.time()
        print('Query -> Term...', end='', flush=True)
        terms_text_t = qp.sentences_to_terms(query[qp.TITLE])
        terms_text_c = qp.sentences_to_terms(query[qp.CONCEPTS])

        terms_id_t = qp.terms_text_to_terms_id(vocab, terms_text_t)
        terms_id_c = qp.terms_text_to_terms_id(vocab, terms_text_c)

        invf_indexes_t = qp.terms_id_to_inverted_file_index(invf_terms, terms_id_t)
        invf_indexes_c = qp.terms_id_to_inverted_file_index(invf_terms, terms_id_c)
        print(f'done. ({time.time()-start_time}sec)')


        '''
        VSM

        - Merge Postings List
        - Create the VS (2D list)
            * i axis: Terms
            * j axis: Merged Postings List
        - Fill the VS with the Inverted File
        '''

        start_time = time.time()
        invf_indexes = list(set(invf_indexes_t + invf_indexes_c))
        merged_postings = vsm.get_merged_postings_list(invf_postings, invf_indexes)

        start_time = time.time()
        print('Creating empty VS...', end='', flush=True)
        VS = vsm.create_zeroed_2D_matrix(len(invf_indexes), len(merged_postings))
        print(f'done. ({time.time()-start_time}sec)')

        start_time = time.time()
        print('Filling VS...', end='', flush=True)
        vsm.fill_matrix(VS, len(invf_indexes), invf_indexes, merged_postings, invf_postings, len(docs))
        print(f'done. ({time.time()-start_time}sec)')


        '''
        Define Query Vector

        - Create a Simple Query Vector
            * Title x10
            * Concept x1
        '''










