import random

inf = open('../data/intuition/combined_4/complete_cleared_all_diag_compress_train_select.csv')
f = inf.readline()

lines = inf.readlines()
inf.close()
trainf = open('../data/intuition/combined_4/complete_cleared_all_diag_compress_train_select_rand.csv','w')
trainf.write(f);
random.shuffle(lines)

for line in lines:
        trainf.write(line)

trainf.close()
