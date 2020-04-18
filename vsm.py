import math
import pathlib
import os
import array
import numpy


def tf_normalization(invf_postings, invf_terms, docs, doclen, avdl, k, b):
    for i in range(len(invf_postings)):
        term = invf_terms[i]
        postings = invf_postings[i]
        for j in range(len(postings)):
            posting_id, tf = postings[j]
            norm = 1 - b + b * doclen[posting_id] / avdl
            iDF = math.log((len(docs)-len(postings)+0.5)/(len(postings)+0.5))
            postings[j] = (posting_id, okapi_bm25(tf, k, norm, iDF))


def okapi_bm25(tf, k, norm, idf):
        return tf * (k+1) / (tf + k*norm) * idf


def create_doc_term_id(docs, invf_terms, invf_postings):
    # doc:list -> term:dict -> tf:float, sorted, for faster tf finding
    doc_term_id = []
    for i in range(len(docs)):
        doc_term_id.append(dict())
    for i in range(len(invf_terms)):
        for posting in invf_postings[i]:
            doc_term_id[posting[0]][i] = posting[1]
    return doc_term_id


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


class VS:
    def __init__(self, doc_term_id_dict):
        self.doc_term_id_dict = doc_term_id_dict
    

    def val(self, i, j):
        try:
            return self.doc_term_id_dict[j][i]
        except KeyError:
            return 0
