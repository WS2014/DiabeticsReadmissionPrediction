inf = open('../data/diabetic_data_normalized_noweight_nopc_noms.csv','r');
outf = open('../data/diabetic_data_normalized_complete_noweight_nopc_noms.csv','w');

flag = 1

for line in inf:
    line1 = line[:-1]
    tok = line1.split(',');
    flag = 1
    for t in tok:
        t = str(t)
        if t == '?':
            flag = 0;
            break;
    
    if flag == 1:
        outf.write(line);
    
inf.close()
outf.close()