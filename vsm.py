import math
import pathlib
import os
import array

import numpy

def get_merged_postings_list(inverted_file_postings, term_indexes):
    merged_postings = set()
    for i in term_indexes:
        for postings in inverted_file_postings[i]:
            merged_postings.add(postings[0])
    merged_postings = list(merged_postings)
    merged_postings.sort()
    return merged_postings


def create_zeroed_2D_matrix(i, j):
    import copy
    A = []
    row = [0.0] * j
    # row = array.array('d', row)
    # print("using list of arrays...", end='')
    for _ in range(i):
        A.append(copy.deepcopy(row))
    print("using list...", end='')
    # A = numpy.array(A)
    # print("using ndarray...", end='')
    # NOTE: Test results suggests approx. same speed
    return A


def idf(df, N, base=10):
    return math.log(N/df, base)
    

def tf_idf(tf, idf):
    return tf * idf


def okapi_tf(cnt, k=100):
    return (k+1) * cnt / (cnt + k)


def okapi_doclen_norm(posting_id, avdl, doclen, b=0.8):
    doclen = doclen[posting_id]
    return 1-b+b*doclen/avdl


def get_avg_doclen(postings, docs):
    sum = 0
    for posting in postings:
        filename = docs[posting]
        path = str(pathlib.Path().absolute())+'/CIRB010/'+filename
        sum += os.path.getsize(path)
    return sum/len(postings)


def fill_matrix(VS, height, inverted_file_term_indexes, merged_postings, inverted_file_postings, docs, avdl, doclen, k=100, b=0.8):
    
    import time
    time_tf = 0
    time_tf_idf = 0
    time_nxt_post = 0

    # Create a dictionary for fast index searching (merged{docID} = index of docID in merged_postings)
    merged = dict()
    for i in range(len(merged_postings)):
        merged[merged_postings[i]] = i
    

    # Rule: weight = tf * idf
    # tf = okapi_tf / okapi_tf_normalization 
    doc_cnt = len(docs)
    for i in range(height):
        postings = inverted_file_postings[inverted_file_term_indexes[i]]

        start_time = time.time_ns()
        for posting in postings:
            # NOTE: 1/3 of time is spent on the Okapi_norm (5s for train_query_1), Solution: precalculated doclen
            time_nxt_post += time.time_ns() - start_time

            start_time = time.time_ns()
            tf = okapi_tf(posting[1], k) / okapi_doclen_norm(posting[0], avdl, doclen, b)
            time_tf += time.time_ns() - start_time

            # NOTE: 2/3 of time is spent on list.index (10s for train_query_1), we use a dict to find the index
            start_time = time.time_ns()
            iDF = idf(len(postings), doc_cnt)
            VS[i][merged[posting[0]]] = tf_idf(tf, iDF)
            # VS[i][merged_postings.index(posting[0])] = tf_idf(tf, iDF)
            time_tf_idf += time.time_ns() - start_time

            start_time = time.time_ns()
    
    print(f"tf_ns:{time_tf/1000000000},idf_ns:{time_tf_idf/1000000000}, nxt_pst_ns:{time_nxt_post/1000000000}...", end='')
            
