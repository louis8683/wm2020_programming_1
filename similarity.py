import math


def dot(VS, qv, invf_indexes, merged_postings):
    result = []
    cnt = 0
    for j in merged_postings:
        cnt += 1
        print(f"\rSimilarity: Dot product {cnt}/{len(merged_postings)}...", end='')
        # print(f"\rComparing posting {cnt}/{len(merged_postings)}...", end='')
        dot = 0
        # Sum(w_q*w_j)
        for i in invf_indexes:
            dot += VS.val(i, j) * qv[i]
        result.append(dot)
    return result


def cosine(VS, qv, invf_indexes, merged_postings, hybrid=False, power=2):
    result = []
    cnt = 1
    for j in merged_postings:
        cnt += 1
        print(f"\r {cnt}/{len(merged_postings)}...", end='', flush=True)
        
        print("#0...", end='', flush=True)
        dot = 0
        # Sum(w_q*w_j)
        for i in invf_indexes:
            dot += VS.val(i, j) * qv[i]
        print("#1...", end='', flush=True)
        # Sum(w_q^2)*Sum(w_j^2)
        wq_sq = 0
        wj_sq = 0
        for i in invf_indexes:
            wq_sq += VS.val(i, j)**2
            wj_sq += qv[i]**2
        if hybrid:
            dot = dot**power
        print("#2...", end='', flush=True)
        cosine = dot/math.sqrt(wq_sq*wj_sq)
        result.append(cosine)
    return result
