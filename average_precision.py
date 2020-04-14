
ans_filename = "./queries/ans_train.csv"
f = open(ans_filename, 'r')
f.readline()
ln = f.readline()
ln = ln.split(',')[1]
ans_ranking = ln.split(' ')
f.close()

my_filename = "./queries/my_ans_train.csv"
f = open(my_filename, 'r')
f.readline()
ln = f.readline()
ln = ln.split(',')[1]
my_ranking = ln.split(' ')
f.close()

# print(ans_ranking)
# print(my_ranking)


def precision(i):
    ans = ans_ranking[:i+1]
    my = my_ranking[:i+1]
    cnt = 0
    for doc in ans:
        try:
            my.index(doc)
            cnt += 1
        except ValueError:
            pass
    return cnt/(i+1)


average_precision = 0
for i in range(len(ans_ranking)):
    average_precision += precision(i)
    # print(precision(i), end='+')

# print(" ")
# print(f"{average_precision}/{len(ans_ranking)}={average_precision/len(ans_ranking)}")
average_precision /= len(ans_ranking)
print(average_precision)
