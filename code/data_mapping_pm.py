inf = open('../data/diabetic_data_normalized_weka.csv','r');
outf = open('../data./diabetic_data_normalized_uniform_weka.csv','w');

ud = 10000

ac = 0
bc = 0
cc = 0 

for line in inf:
    line1 = line[:-1]
    tok = line1.split(',');
    d1 = tok[-1]
    
    if d1 == 'NO':
        if bc < ud:
            bc = bc + 1
            outf.write(line)
            
    if d1 == '>30':
        if ac < ud:
            ac = ac + 1
            outf.write(line)
            
    if d1 == '<30':
        if cc < ud:
            cc = cc + 1
            outf.write(line)
            
inf.close()
outf.close()