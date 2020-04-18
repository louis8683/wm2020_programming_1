import math
import copy
import time
import sys
import pathlib

import numpy

import query_processing as qp
import vsm
import vsm_io


'''
Rules are marked with 'Rule:' in the comments
'''


mode = 'train' # 'train' or 'test
use_sys_arguments = False
should_read_models = False
should_tf_normalization = should_read_models
should_do_rocchio = True
one_query_only = False
test_ap = one_query_only
test_map = not test_ap

avdl = 2520 # preprocessed avdl

# varibles
# Okapi
k, b = 2, 1
# Rocchio
alpha, beta = 1, 0.8
num_relevant_docs = 1000
num_expanded_terms = 5000
# Similarity
method = 'dot'

if __name__ == "__main__":


    '''
    Parse Command Line Arguments

    - "-r": If specified, turn on the relevance feedback
    - "-i query-file": The input query file
    - "-o ranked-list": The output ranked list file
    - "-m model-dir": The input model directory (includes three model files)
    - "-d NTCIR-dir": The directory of NTCIR documents
    '''


    if use_sys_arguments:
        rel_on, input_query_filename, output_ranked_filename, model_dir, ntcir_dir = vsm_io.parse_sys_arguments(sys.argv)
        this_program_file_dir = str(pathlib.Path(__file__).parent.absolute())
        print("Program Directory: ", this_program_file_dir)
    else:
        rel_on = should_do_rocchio
        input_query_filename = f"./queries/query-{mode}.xml"
        output_ranked_filename = f"./queries/my_ans_{mode}.csv"
        model_dir = "./model"
        ntcir_dir = "./CIRB010"
        this_program_file_dir = "./"


    '''
    Read in the models.

    - Read Vocabulary (vocab.all) into a list
    - Read Documents (file-list) into a list
    - Read Inverted File (inverted-file) into two lists, one for 'terms' and one for 'postings'
    '''


    start_time = time.time()
    print('Reading models...', end='', flush=True)
    if should_read_models:
        vocab = vsm_io.read_vocab(model_dir + "/vocab.all")

        docs = vsm_io.read_docs(model_dir + "/file-list")

        doclen = vsm_io.read_doclen(this_program_file_dir + "/file-len")
    
        invf_terms, invf_postings = vsm_io.read_inverted_file(model_dir + "/inverted-file")
        
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


    if should_tf_normalization:
        start_time = time.time()
        print('TF Normalization...', end='', flush=True)
        vsm.tf_normalization(invf_postings, invf_terms, docs, doclen, avdl, k, b)        
        doc_term_id = vsm.create_doc_term_id(docs, invf_terms, invf_postings)
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
    f = open(output_ranked_filename, "w")
    f.write("query_id,retrieved_docs\n")

    queries = qp.read_queries(input_query_filename)

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

        
        start_time = time.time()
        print('Merging...', end='', flush=True)
        merged_postings = vsm.get_merged_postings_list(invf_postings, invf_indexes)
        print(f'done. ({time.time()-start_time}sec)')

        print('Creating VS Object...', end='', flush=True)
        VS = vsm.VS(doc_term_id)
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
        print(f'done. ({time.time()-start_time}sec)')

        if should_do_rocchio:
            # Select Relevant Documents
            relevant_postings = []
            # Similarity
            if method == 'dot':
                sim = similarity.dot(VS, query_vec, invf_indexes, merged_postings)
            else:
                sim = similarity.cosine(VS, query_vec, invf_indexes, merged_postings, hybrid=True, power=2)
            # Ranking
            rank = []
            for i in range(len(sim)):
                rank.append((sim[i], merged_postings[i]))
            rank.sort(reverse=True)
            if num_relevant_docs > len(rank):
                num_relevant_docs = len(rank)
            for i in range(num_relevant_docs):
                relevant_postings.append(rank[i][1])

            start_time = time.time()
            print('Rocchio...', end='', flush=True)
            qp.rocchio(VS, query_vec, invf_indexes, relevant_postings, doc_term_id, alpha=alpha, beta=beta)
            
            # Expand Query
            invf_indexes = qp.expand_query(query_vec, cutoff=num_expanded_terms)
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
            sim = similarity.cosine(VS, query_vec, invf_indexes, merged_postings, hybrid=True, power=2)
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

    if not use_sys_arguments:
        if test_ap:
            print("Average Precision")
            exec(open(this_program_file_dir + "/average_precision.py").read())    

        if test_map:
            print("Mean Average Precision")
            exec(open(this_program_file_dir + "/mean_average_precision.py").read())    
