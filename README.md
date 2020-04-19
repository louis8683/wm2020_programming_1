# wm2020_programming_1

Web Retrival and Web Mining, 2020 Spring
Programming Assignment 1 (Due 4/20)

Strong Baseline: 0.73184
Best result on Kaggle competition: 0.78541 (public rank: 18/87)

How to run (python):
$python hw1_vsm_rocchio.py
How to run (shell script):
$./execute.sh -r -i [query-file] -o [ranked-list] -m [model-dir] -d [NTCIR-dir]

Parameters:
Okapi/BM25: k=2, b=1
Rocchio: alpha=1, beta=0.8
Others: # relevant documents=10000, # expanded terms=100 