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


mode = 'train' # 'train' or 'test
should_read_models = False
should_tf_normalization = should_read_models
should_create_vs = True
test_ap = True
test_map = False
one_query_only = False

avdl = 2520 # preprocessed avdl

# varibles
# Okapi
k, b = 1.2, 0.75
# Rocchio
alpha, beta = 1, 0.8
# Similarity
method = 'dot'

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
        
        print("Creating doc-term...", end='')
        
        # Dicts for fast index finding
        # term->vocab index
        vocab_index = dict()
        for i in range(len(vocab)):
            vocab_index[vocab[i]] = i

        # term->invf_index
        invf_terms_index = dict()
        for i in range(len(invf_terms)):
            invf_terms_index[invf_terms[i]] = i
        
    print(f'done. ({time.time()-start_time}sec)')

    
    '''
    TF Normalization
    '''


    def okapi_bm25(tf, k, norm, idf):
        return tf * (k+1) / (tf + k*norm) * idf

    
    if should_tf_normalization:
        start_time = time.time()
        print('TF Normalization...', end='', flush=True)
        
        for i in range(len(invf_postings)):
            term = invf_terms[i]
            postings = invf_postings[i]
            for j in range(len(postings)):
                posting_id, tf = postings[j]
                norm = 1 - b + b * doclen[posting_id] / avdl
                iDF = math.log((len(docs)-len(postings)+0.5)/(len(postings)+0.5))
                if i % 1000 == 0 and j % 1000 == 0 and tf % 3 == 0:
                    print(f'\rTF Normalization...tf:{tf} norm:{norm} idf:{iDF} okapi:{okapi_bm25(tf, k, norm, iDF)}...', end='\n', flush=True)
                postings[j] = (posting_id, okapi_bm25(tf, k, norm, iDF))
        
        # doc:list -> term:dict -> tf:float, sorted, for faster tf finding
        doc_term_id = []
        for i in range(len(docs)):
            doc_term_id.append(dict())
        for i in range(len(invf_terms)):
            for posting in invf_postings[i]:
                doc_term_id[posting[0]][i] = posting[1]
        
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
        invf_indexes = list(set(invf_indexes_t + invf_indexes_c))
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
            print('Merging...', end='', flush=True)
            merged_postings = vsm.get_merged_postings_list(invf_postings, invf_indexes)
            # merged_invf_indexes = vsm.get_merged_invf_indexes(merged_postings, doc_term_id)
            # merged_invf_indexes.sort()
            print(f'done. ({time.time()-start_time}sec)')

            VS = vsm.VS(invf_terms, invf_postings, docs, doclen, doc_term_id,avdl, k, b)
            #print('Creating Empty VS...', end='', flush=True)
            #VS = vsm.create_zeroed_2D_matrix(len(merged_invf_indexes), len(merged_postings))
            #print(f'done. ({time.time()-start_time}sec)')

            #start_time = time.time()
            #print('Filling VS...', end='', flush=True)
            #vsm.fill_matrix(VS, len(merged_invf_indexes), merged_invf_indexes, merged_postings, invf_postings, docs, avdl, doclen, k=k, b=b)
            print(f'done. ({time.time()-start_time}sec)')
        


        '''
        Define Query Vector

        - Create a Unit Vector
        - Add Weight on title
            * Title x5? (X)
        '''

        
        start_time = time.time()
        print('Creating the Query Vector...', end='', flush=True)
        query_vec = qp.query_vector(invf_indexes, len(invf_terms))
        #qp.weighted_title(query_vec, invf_indexes_t, invf_indexes, weight=1)
        # NOTE: Generally doesn't improve results
        print(f'done. ({time.time()-start_time}sec)')

        start_time = time.time()
        print('Rocchio...', end='', flush=True)
        qp.rocchio(VS, query_vec, invf_indexes, merged_postings, alpha=alpha, beta=beta)
        # NOTE: How can we define "relevant" docs? Now: Avg Cosine
        
        # Expand Query
        invf_indexes = []
        for i in range(len(query_vec)):
            if query_vec[i] != 0:
                invf_indexes.append(i)
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
        if method == 'dot':
            sim = similarity.dot(VS, query_vec, invf_indexes, merged_postings)
        else:
            sim = similarity.cosine(VS, query_vec, hybrid=True, power=2)
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
        for r in rank[:10]:
            print(r)

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










