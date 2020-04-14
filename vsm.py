import math


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
    row = [0] * j
    for _ in range(i):
        A.append(copy.deepcopy(row))
    return A


def idf(df, N, base=10):
    return math.log(N/df, base)
    

def tf_idf(tf, idf):
    return tf * idf


def fill_matrix(VS, height, inverted_file_term_indexes, merged_postings, inverted_file_postings, doc_cnt):
    # Rule: weight = tf * idf
    for i in range(height):
        postings = inverted_file_postings[inverted_file_term_indexes[i]]
        for posting in postings:
            VS[i][merged_postings.index(posting[0])] = tf_idf(posting[1], idf(len(postings), doc_cnt, base=10))