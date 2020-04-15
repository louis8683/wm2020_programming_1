# This file is for testing!
import pathlib
import os


def avdl():
    sum = 0
    cnt = 0
    for filename in docs:
        path = str(pathlib.Path().absolute())+'/CIRB010/'+filename
        sum += os.path.getsize(path)
        cnt += 1
        print(f'\r(processed {cnt}/{len(docs)})', end='')
    avdl = sum / len(docs)

    print("\nAVDL:", avdl)

    # avdl = 2520


print("file-list -> file-len")
f_w = open("file-len", "w")
f_r = open("./model/file-list", "r")

ln = f_r.readline()
cnt = 0
while ln != "":
    filename = ln[:-1]
    path = str(pathlib.Path().absolute())+'/CIRB010/'+filename
    doclen = os.path.getsize(path)
    f_w.write(f"{doclen}\n")
    ln = f_r.readline()
    cnt += 1
    print(f"\rProcessed {cnt} files", end='')
print(" ")

f_w.close()
f_r.close()
