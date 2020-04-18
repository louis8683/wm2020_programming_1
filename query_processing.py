import math


NUMBER = 0
TITLE = 1
CONCEPTS = 2


def read_queries(filename):
        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        root = tree.getroot()
        queries = []
        for topic in root:
            # Every topic
            number, title, question, narrative, concepts = topic
            query_id = number.text[-3:]
            title = title.text
            concepts = concepts.text.split("\n")[1].split("。")[0].split("、")
            queries.append((query_id, title, concepts))
        return queries


def sentences_to_terms(sentences, unigram=False):
    terms_text = []
    for sentence in sentences:
        if len(sentence) <= 2:
            terms_text.append(sentence)
        else:
            for i in range(len(sentence)-1):
                terms_text.append(sentence[i:i+2])
                if unigram:
                    terms_text.append(sentence[i:i+1])
            if unigram:
                terms_text.append(sentence[-1:])
    # TODO: modify this to add weight with occurrence?
    return list(set(terms_text))


def terms_text_to_terms_id(vocab_index, terms):
    terms_id = []
    for term in terms:
        # Translate text to id in vocab.all
        id_1 = vocab_index[term[0]]
        if len(term) == 1:
            id_2 = -1
        else:
            id_2 = vocab_index[term[1]]
        terms_id.append((id_1,id_2))
    return terms_id


def terms_id_to_inverted_file_index(inverted_file_terms_index, terms_id):
    inverted_indexes = []
    for term in terms_id:
        # Find inverted-file index of each term and remove the ones not in vocab.all
        try:
            index = inverted_file_terms_index[term]
            inverted_indexes.append(index)
        except KeyError:
            print(f'no term {term}')
            terms_id.remove(term)
    return inverted_indexes


'''
Functions for the Query Vector
'''


def query_vector(invf_indexes_rel, term_cnt):
    print(f"rel cnt={len(invf_indexes_rel)}...", end='')
    qv = [0] * term_cnt
    for ind in invf_indexes_rel:
        qv[ind] = 1
    return qv


def rocchio(VS, qv, invf_indexes, merged_postings, doc_term_id, alpha=1, beta=0.8): #, gamma=0.1, method='cos', threshold='avg'):
    '''
    The REAL Rocchio Feedback (Without gamma)

    Rocchio: qv' = alpha*qv + beta/|D|*sum(dv) - gamma/|D|*sum(dv)
    We consider relevant documents only
    qv' = alpha * qv + beta * sum(dv)/|D|
    
    Q: How do we define "Relevant" and "Not Relevant"?
    - Proposed Metric: Cosine or Dot > threshold t
    '''
    
    import time
    lookup_time = 0

    if beta == 0: # No weight
        return

    relevant_terms = set() # For Performance

    # sum(dv_r)
    print(" ")
    sum_dv_r = [0] * len(qv)
    cnt = 0
    for j in merged_postings: # for every relevant document
        for i in doc_term_id[j].keys(): # For every relevant term
            start_time = time.time_ns()
            sum_dv_r[i] += VS.val(i,j)
            relevant_terms.add(i)
            lookup_time += time.time_ns() - start_time
        print(f"\rLookup Time: {lookup_time/1000000000}...", end='', flush=True)
    print(" ")


    # find |Dr|
    dr_length = len(merged_postings)

    # Exceptions with division 0
    if dr_length == 0:
        beta, dr_length = 0, 1
        print("(Rocchio: No Relevant Documents)...", end='')
    
    # qv' = alpha * qv + beta * sum(dv_r)/|Dr| - gamma * sum(dv_n)/|Dn|
    cnt = 0
    for i in relevant_terms:
        cnt += 1
        print(f"\rRocchio, processing term {cnt}/{len(relevant_terms)}...", end='', flush=True)
        qv[i] = alpha * qv[i] + beta * sum_dv_r[i] / dr_length


def expand_query(qv, cutoff=100):
    import copy
    rank = copy.deepcopy(qv)
    rank.sort(reverse=True)
    if len(rank) > cutoff:
        cutoff_val = rank[cutoff]
    else:
        return

    invf_indexes = []
    for i in range(len(qv)):
        # NOTE: Need a rule on expanding query, e.g., threshold value
        if qv[i] > cutoff_val:
            invf_indexes.append(i)
    print(f"new query length {len(invf_indexes)}...", end='')
    return invf_indexes
    