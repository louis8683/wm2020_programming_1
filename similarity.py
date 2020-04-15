import math


def dot(VS, qv):
    result = []
    for j in range(len(VS[0])):
        dot = 0
        # Sum(w_q*w_j)
        for i in range(len(qv)):
            dot += VS[i][j] * qv[i]
        result.append(dot)
    return result


def cosine(VS, qv, hybrid=True, power=2):
    result = []
    for j in range(len(VS[0])):
        dot = 0
        cosine = 0
        # Sum(w_q*w_j)
        for i in range(len(qv)):
            dot += VS[i][j] * qv[i]
        # Sum(w_q^2)*Sum(w_j^2)
        wq_sq = 0
        wj_sq = 0
        for i in range(len(qv)):
            wq_sq += VS[i][j] * VS[i][j]
            wj_sq += qv[i] * qv[i]
        if hybrid:
            dot = dot**power
        cosine = dot/math.sqrt(wq_sq*wj_sq)
        result.append(cosine)
    return result


def euclidean(VS, qv):
    result = []
    for j in range(len(VS[0])):
        euclidean = 0
        # Sum((w_q-w_j)^2)
        for j in range(len(qv)):
            euclidean += (VS[i][j] - qv[i]) * (VS[i][j] - qv[i])
        euclidean = math.sqrt(euclidean)
        result.append(euclidean)
    return result
