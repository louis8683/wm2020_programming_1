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


def get_merged_invf_indexes(merged_postings, doc_term_id):
    # invf_indexes = []
    invf_indexes = set()
    cnt = 0
    merged_postings_sorted_by_term_cnt = []
    for posting in merged_postings:
        merged_postings_sorted_by_term_cnt.append((len(doc_term_id[posting]), posting))
    merged_postings_sorted_by_term_cnt.sort()
    for posting in merged_postings_sorted_by_term_cnt:
        posting = posting[1]
        cnt += 1
        print(f"\rmerged {cnt}/{len(merged_postings_sorted_by_term_cnt)} postings...", end='')
        j = 0
        print(f"posting terms:{len(doc_term_id[posting])}...", end='')
        for i in range(len(doc_term_id[posting])):
            '''
            if i >= len(invf_indexes) or j >= len(invf_indexes) or doc_term_id[posting][i] < invf_indexes[j]:
                invf_indexes.insert(j, doc_term_id[posting][i])
                j += 1
            while j < len(invf_indexes) and doc_term_id[posting][i] >= invf_indexes[j]:
                j += 1
            '''
            invf_indexes.add(doc_term_id[posting][i])
        print(f"term cnt:{len(invf_indexes)}...", end='')
    invf_indexes = list(invf_indexes)
    invf_indexes.sort()
    return invf_indexes


def create_zeroed_2D_matrix(i, j):
    import copy
    A = []
    #row = [0.0] * j
    #row = array.array('d', row)
    #cnt = 0
    #for _ in range(i):
    #    cnt += 1
    #    print(f"\rusing list of arrays...{cnt}/{i} rows created...", end='')
    #    A.append(copy.deepcopy(row))
    #print("using list...", end='')
    A = numpy.ndarray((i,j), dtype=float)
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

    # Create a dictionary for fast index searching (merged{docID} = index of docID in merged_postings)
    merged = dict()
    for i in range(len(merged_postings)):
        merged[merged_postings[i]] = i
    

    # Rule: weight = tf * idf
    # tf = okapi_tf / okapi_tf_normalization 
    doc_cnt = len(docs)
    for i in range(height):
        postings = inverted_file_postings[inverted_file_term_indexes[i]]
        for posting in postings:
            # NOTE: 1/3 of time is spent on the Okapi_norm (5s for train_query_1), Solution: precalculated doclen
            tf = okapi_tf(posting[1], k) / okapi_doclen_norm(posting[0], avdl, doclen, b)
            
            # NOTE: 2/3 of time is spent on list.index (10s for train_query_1), we use a dict to find the index
            iDF = idf(len(postings), doc_cnt)
            VS[i][merged[posting[0]]] = tf_idf(tf, iDF)

            avg_raw_tf += posting[1]
            if posting[1] > max_raw_tf:
                max_raw_tf = posting[1]
            if posting[1] > cutoff:
                cnt_over += 1
            cnt += 1


class VS:
    def __init__(self, inverted_file_term, inverted_file_postings, docs, doclen, doc_term_id_dict, avdl, k=100, b=0.8):
        self.invf_terms = inverted_file_term
        self.invf_postings = inverted_file_postings
        self.docs = docs
        self.doclen = doclen
        self.doc_term_id_dict = doc_term_id_dict
        self.avdl = avdl
        self.k = k
        self.b = b
    

    def val(self, i, j):
        try:
            return self.doc_term_id_dict[j][i]
        except KeyError:
            return 0