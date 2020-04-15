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


def unit_vector(size):
    return [math.sqrt(1/size)] * size


def weighted_title(qv, title_indexes ,index_list, weight=5, mode='*'):
    for title in title_indexes:
        try:
            i = index_list.index(title)
            if mode == '*':
                qv[i] *= weight
            elif mode == '+':
                qv[i] += weight
        except ValueError:
            pass


def _cosine(VS, qv, col):
    dot = 0
    cosine = 0
    # Sum(w_q*w_j)
    for i in range(len(qv)):
        dot += float(VS[i][col]) * qv[i]
    # Sum(w_q^2)*Sum(w_j^2)
    wq_sq = 0
    wj_sq = 0
    for i in range(len(qv)):
        wq_sq += float(VS[i][col]) * float(VS[i][col])
        wj_sq += qv[i] * qv[i]
    cosine = dot/math.sqrt(wq_sq*wj_sq)
    return cosine


def _dot(VS, qv, col):
    dot = 0
    # Sum(w_q*w_j)
    for i in range(len(qv)):
        dot += VS[i][col] * qv[i]
    return dot



def rocchio_feedback(VS, qv, alpha=0.8, beta=0.2, gamma=0.2, similarity='dot', threshold='avg'):
    '''
    Rocchio: qv' = alpha*qv + beta/|D|*sum(dv) - gamma/|D|*sum(dv)
    We consider relevant documents only
    qv' = alpha * qv + beta * sum(dv)/|D|
    
    Q: How do we define "Relevant" and "Not Relevant"?
    - Proposed Metric: Cosine or Dot > threshold t
    '''
    # find threshold
    similarity = []
    sum_sim = 0
    for i in range(len(VS[0])):
        sim = _cosine(VS, qv, i)
        if similarity == 'dot':
            sim = _dot(VS, qv, i)
        similarity.append(sim)
        sum_sim += sim
    avg_sim = sum_sim/len(VS[0])

    if threshold == 'avg':
        t = avg_sim
        print(f"(thres={t})...", end='')
    else:
        t = threshold
        print(f"([d]thres={t})...", end='')
    
    # Construct relevance list
    relevant = [] # True: relevant, False: not relevant
    relevant_cnt = 0
    for i in range(len(VS[0])):
        if similarity[i] > t:
            relevant.append(True)
            relevant_cnt += 1
        else:
            relevant.append(False)
    
    # sum(dv_r), sum(dv_n)
    sum_dv_r = [0] * len(qv)
    sum_dv_n = [0] * len(qv)
    for j in range(len(VS[0])): # for every relevant document
        if relevant[j]:
            for i in range(len(qv)):
                sum_dv_r[i] += VS[i][j]
        else:  
            for i in range(len(qv)):
                sum_dv_n[i] += VS[i][j]
    
    # find |Dr|
    dr_length = 0
    for i in range(len(sum_dv_r)):
        dr_length += sum_dv_r[i] * sum_dv_r[i]
    dr_length = math.sqrt(dr_length)
    # find |Dn|
    dn_length = 0
    for i in range(len(sum_dv_n)):
        dn_length += sum_dv_n[i] * sum_dv_n[i]
    dn_length = math.sqrt(dn_length)

    # Exceptions with division 0
    if dr_length == 0:
        beta, dr_length = 0, 1
        print("(Rocchio: No Relevant Documents)...", end='')
    if dn_length == 0:
        gamma, dn_length = 0, 1
        print("(Rocchio: No Non-Relevant Documents)...", end='')
    
    # qv' = alpha * qv + beta * sum(dv_r)/|Dr| - gamma * sum(dv_n)/|Dn|
    for i in range(len(qv)):
        qv[i] = alpha*qv[i] + beta * sum_dv_r[i] / dr_length - gamma * sum_dv_n[i] / dn_length
