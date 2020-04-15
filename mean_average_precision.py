ans_filename = "./queries/ans_train.csv"
f = open(ans_filename, 'r')
ln = f.readline()
ln = f.readline()
ans_rankings = []
while ln != "":
    ln = ln.split(',')[1]
    ans_ranking = ln.split(' ')
    ans_rankings.append(ans_ranking)
    ln = f.readline()
f.close()

my_filename = "./queries/my_ans_train.csv"
f = open(my_filename, 'r')
ln = f.readline()
ln = f.readline()
my_rankings = []
while ln != "":
    ln = ln.split(',')[1]
    my_ranking = ln.split(' ')
    my_rankings.append(my_ranking)
    ln = f.readline()
f.close()


def precision(ans_ranking, my_ranking):
    cnt = 0
    for doc in ans_ranking:
        try:
            my_ranking.index(doc)
            cnt += 1
        except ValueError:
            pass
    return cnt/len(ans_ranking)


sum_ap = 0
for i in range(len(ans_rankings)):
    average_precision = 0
    ans_ranking = ans_rankings[i]
    my_ranking = my_rankings[i]
    for j in range(len(ans_ranking)):
        average_precision += precision(ans_ranking[:j+1], my_ranking[:j+1])
    average_precision /= len(ans_ranking)
    sum_ap += average_precision
    print(average_precision)
mAP = sum_ap / len(ans_rankings)
print("mAP = ", mAP)
