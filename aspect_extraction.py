import pandas as pd
pd.set_option('display.width', 2000)
import numpy as np
import csv
import os
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'..\coreNLP\stanford-corenlp-full-2018-02-27')

def read_file_as_str(file_path):
    if not os.path.isfile(file_path):
        raise TypeError(file_path + " does not exist")
    all_the_text = open(file_path).read()
    return all_the_text

import datetime
s = datetime.datetime.now()

### BOOK
### BOOK
### BOOK
print("---------- book start---- book start--- book start---")
for i in range(0, 3000): 
    num = 0
    file = read_file_as_str('../Book/pos/%u.txt' % i)
    pos_tag = nlp.pos_tag(file)
    for pos in pos_tag:
        if pos[1] == 'NN':
            num += 1
            with open('../Book/pos_aspect/%u.txt' % i, "a") as f:
                f.write(pos[0])
                f.write(' ')
    if num == 0:
        with open('../Book/pos_aspect/%u.txt' % i, "a") as f:
            f.write('null')
            print("%u is no target", i)



for i in range(0, 3000): 
    num = 0
    file = read_file_as_str('../Book/neg/%u.txt' % i)
    pos_tag = nlp.pos_tag(file)
    for pos in pos_tag:
        if pos[1] == 'NN':
            num += 1
            with open('../Book/neg_aspect/%u.txt' % i, "a") as f:
                f.write(pos[0])
                f.write(' ')
    if num == 0:
        with open('../Book/neg_aspect/%u.txt' % i, "a") as f:
            f.write('null')
            print("%u is no target", i)

### DVD
### DVD
### DVD
print("---------- dvd start---- dvd start--- dvd start---")

for i in range(0, 3000): 
    num = 0
    file = read_file_as_str('../DVD/pos/%u.txt' % i)
    pos_tag = nlp.pos_tag(file)
    for pos in pos_tag:
        if pos[1] == 'NN':
            num += 1
            with open('../DVD/pos_aspect/%u.txt' % i, "a") as f:
                f.write(pos[0])
                f.write(' ')
    if num == 0:
        with open('../DVD/pos_aspect/%u.txt' % i, "a") as f:
            f.write('null')
            print("file txt %u is no target", i)


for i in range(0, 3000): 
    num = 0
    file = read_file_as_str('../DVD/neg/%u.txt' % i)
    pos_tag = nlp.pos_tag(file)
    for pos in pos_tag:
        if pos[1] == 'NN':
            num += 1
            with open('../DVD/neg_aspect/%u.txt' % i, "a") as f:
                f.write(pos[0])
                f.write(' ')
    if num == 0:
        with open('../DVD/neg_aspect/%u.txt' % i, "a") as f:
            f.write('null')
            print("file txt %u is no target", i)

### Electronics
### Electronics
### Electronics
print("---------- ele start---- ele start--- ele start---")
for i in range(0, 3000): 
    num = 0
    file = read_file_as_str('../Electronics/pos/%u.txt' % i)
    pos_tag = nlp.pos_tag(file)
    for pos in pos_tag:
        if pos[1] == 'NN':
            num += 1
            with open('../Electronics/pos_aspect/%u.txt' % i, "a") as f:
                f.write(pos[0])
                f.write(' ')
    if num == 0:
        with open('../Electronics/pos_aspect/%u.txt' % i, "a") as f:
            f.write('null')
            print("file txt %u is no target", i)


for i in range(0, 3000): 
    num = 0
    file = read_file_as_str('../Electronics/neg/%u.txt' % i)
    pos_tag = nlp.pos_tag(file)
    for pos in pos_tag:
        if pos[1] == 'NN':
            num += 1
            with open('../Electronics/neg_aspect/%u.txt' % i, "a") as f:
                f.write(pos[0])
                f.write(' ')
    if num == 0:
        with open('../Electronics/neg_aspect/%u.txt' % i, "a") as f:
            f.write('null')
            print("file txt %u is no target", i)

### Kitchen
### Kitchen
### Kitchen
print("---------- kitchen start---- kitchen start--- kitchen start---")
for i in range(0, 3000): 
    num = 0
    file = read_file_as_str('../Kitchen/pos/%u.txt' % i)
    pos_tag = nlp.pos_tag(file)
    for pos in pos_tag:
        if pos[1] == 'NN':
            num += 1
            with open('../Kitchen/pos_aspect/%u.txt' % i, "a") as f:
                f.write(pos[0])
                f.write(' ')
    if num == 0:
        with open('../Kitchen/pos_aspect/%u.txt' % i, "a") as f:
            f.write('null')
            print("file txt %u is no target", i)


for i in range(0, 3000): 
    file = read_file_as_str('../Kitchen/neg/%u.txt' % i)
    num = 0
    pos_tag = nlp.pos_tag(file)
    for pos in pos_tag:
        if pos[1] == 'NN':
            num += 1
            with open('../Kitchen/neg_aspect/%u.txt' % i, "a") as f:
                f.write(pos[0])
                f.write(' ')
    if num == 0:
        with open('../Kitchen/neg_aspect/%u.txt' % i, "a") as f:
            f.write('null')
            print("file txt %u is no target", i)


nlp.close()
e = datetime.datetime.now()

print("Time last : ", e-s)