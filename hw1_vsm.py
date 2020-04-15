import math
import copy
import time
import array

import numpy

import query_processing as qp
import vsm
import vsm_io


'''
Rules are marked with 'Rule:' in the comments
'''


mode = 'train'
should_read_models = False
should_create_vs = True
test_ap = True
test_map = True
one_query_only = False

avdl = 2520 # preprocessed avdl

# varibles
# Okapi
k, b = 20, 0.9
# Rocchio
alpha, beta, gamma, similarity = 1, 0.8, 0.1, 'cos'

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
        vocab = vsm_io.read_vocab("./model/vocab.all")

        docs = vsm_io.read_docs("./model/file-list")

        doclen = vsm_io.read_doclen("file-len")
    
        invf_terms, invf_postings = vsm_io.read_inverted_file("./model/inverted-file")
    
        vocab_index = dict()
        for i in range(len(vocab)):
            vocab_index[vocab[i]] = i

        invf_terms_index = dict()
        for i in range(len(invf_terms)):
            invf_terms_index[invf_terms[i]] = i
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


    # Open the file for writing ranks    
    f = open(f"./queries/my_ans_{mode}.csv", "w")
    f.write("query_id,retrieved_docs\n")

    queries = qp.read_queries(f"./queries/query-{mode}.xml")

    cnt = 0
    for query in queries:
        cnt += 1
        print(f"Query Title {cnt}: ", query[qp.TITLE])
        
        start_time = time.time()
        print('Query -> Term...', end='', flush=True)
        terms_text_t = qp.sentences_to_terms(query[qp.TITLE])
        terms_text_c = qp.sentences_to_terms(query[qp.CONCEPTS])

        terms_id_t = qp.terms_text_to_terms_id(vocab_index, terms_text_t)
        terms_id_c = qp.terms_text_to_terms_id(vocab_index, terms_text_c)

        invf_indexes_t = qp.terms_id_to_inverted_file_index(invf_terms_index, terms_id_t)
        invf_indexes_c = qp.terms_id_to_inverted_file_index(invf_terms_index, terms_id_c)
        print(f'done. ({time.time()-start_time}sec)')


        '''
        VSM

        - Merge Postings List
        - Create the VS (2D array)
            * i axis: Terms
            * j axis: Merged Postings List
        - Fill the VS with the Inverted File
        '''

        
        if should_create_vs:
            start_time = time.time()
            print('Creating Empty VS...', end='', flush=True)
            invf_indexes = list(set(invf_indexes_t + invf_indexes_c))
            merged_postings = vsm.get_merged_postings_list(invf_postings, invf_indexes)

            VS = vsm.create_zeroed_2D_matrix(len(invf_indexes), len(merged_postings))
            print(f'done. ({time.time()-start_time}sec)')

            start_time = time.time()
            print('Filling VS...', end='', flush=True)
            vsm.fill_matrix(VS, len(invf_indexes), invf_indexes, merged_postings, invf_postings, docs, avdl, doclen, k=k, b=b)
            print(f'done. ({time.time()-start_time}sec)')
        


        '''
        Define Query Vector

        - Create a Unit Vector
        - Add Weight on title
            * Title x5? (X)
        '''

        
        start_time = time.time()
        print('Creating the Query Vector...', end='', flush=True)

        query_vec = qp.unit_vector(len(invf_indexes))

        # qp.weighted_title(query_vec, invf_indexes_t, invf_indexes, weight=1)
        # NOTE: Generally doesn't improve results

        qp.rocchio_feedback(VS, query_vec, alpha=alpha, beta=beta, gamma=gamma, similarity=similarity)
        # NOTE: How can we define "relevant" docs? Now: Avg Cosine

        print(f'done. ({time.time()-start_time}sec)')

        
        '''
        Similarity

        - Define a Similarity Function 'sim'
            * Dot
            * Cosine
            * Enclidean
            * Hybrid? (Dot + Cosine)
        - Calculate Every sim(qv,D) and store results
        '''

        start_time = time.time()
        print('Calculating Similarity...', end='', flush=True)
        import similarity
        # sim = similarity.cosine(VS, query_vec, hybrid=True, power=2)
        sim = similarity.dot(VS, query_vec)
        print(f'done. ({time.time()-start_time}sec)')


        '''
        Ranking

        - Sort by the Similarity 
            - Create a tuple with (similarity, doc_name)
            - Sort
        - Extract the 100 most relevant docs
        '''

        rank = []
        for i in range(len(sim)):
            rank.append((sim[i], docs[merged_postings[i]].split('/')[-1].lower()))
        rank.sort(reverse=True)

        # Print ranking
        # for r in rank[:10]:
        #    print(r)

        max_rank = 100
        rank = rank[:max_rank]
        for i in range(len(rank)):
            rank[i] = rank[i][1]
        vsm_io.write_rank(f, query[qp.NUMBER], rank)

        
        if one_query_only:
            break # For faster debugging
        
        print(" ")
    
    f.close()

    if test_ap:
        print("Average Precision")
        exec(open("average_precision.py").read())    

    if test_map:
        print("Mean Average Precision")
        exec(open("mean_average_precision.py").read())    










