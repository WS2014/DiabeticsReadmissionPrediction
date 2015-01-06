import random

inf = open('../data/intuition/complete_cleared_all_feature_test.csv')
#inf = open('../data/intuition/complete_cleared_all_feature.csv')
#inf = open('../data/diabetic_data_normalized.csv')
a = inf.readline()
lines = inf.readlines()
inf.close()
out = open('../data/intuition/complete_cleared_all_diag_compress_test.csv.csv','w')
#out = open('../data/intuition/complete_cleared_all_diag_compress.csv','w')
#out = open('../data/diabetic_data_normalized_all_diag_compress.csv','w')

out.write(a);
map = dict()
map['A'] = (1,139)
map['B'] = (140,239)
map['C'] = (240,279)
map['D'] = (280,289)
map['E'] = (290,319)
map['F'] = (320,359)
map['G'] = (360,389)
map['H'] = (390,459)
map['I'] = (460,519)
map['J'] = (520,579)
map['K'] = (580,629)
map['L'] = (630,679)
map['M'] = (680,709)
map['N'] = (710,739)
map['O'] = (740,759)
map['P'] = (760,779)
map['Q'] = (780,799)
map['R'] = (800,999)
map['X'] = "E"
map['Y'] = "V"

for line in lines:
    row = line[:-1]
    row = str(row)
    tok = row.split(',')
    new_tok = tok
    #d1,d2,d3
    for i in range(13,16):
    #for i in range(18,21):
        sd = tok[i]
        if 'E' in sd or 'V' in sd or '?' in sd:
            d = str(sd)
            if 'E' in d:
                new_tok[i] = 'X'
            elif 'V' in d:
                new_tok[i] = 'Y'
        else:
            #print sd
            d = float(sd)
            
            if d >= 1 and d <= 139 :
                new_tok[i] = 'A'
            elif d >= 140 and d <= 239 :
                new_tok[i] = 'B'
            elif d >= 240 and d <= 279 :
                new_tok[i] = 'C'
            elif d >= 280 and d <= 289 :
                new_tok[i] = 'D'
            elif d >= 290 and d <= 319 :
                new_tok[i] = 'E'
            elif d >= 320 and d <= 359 :
                new_tok[i] = 'F'
            elif d >= 360 and d <= 389 :
                new_tok[i] = 'G'
            elif d >= 390 and d <= 459 :
                new_tok[i] = 'H'
            elif d >= 460 and d <= 519 :
                new_tok[i] = 'I'
            elif d >= 520 and d <= 579 :
                new_tok[i] = 'J'
            elif d >= 580 and d <= 629 :
                new_tok[i] = 'K'
            elif d >= 630 and d <= 679 :
                new_tok[i] = 'L'
            elif d >= 680 and d <= 709 :
                new_tok[i] = 'M'
            elif d >= 710 and d <= 739 :
                new_tok[i] = 'N'
            elif d >= 740 and d <= 759 :
                new_tok[i] = 'O'
            elif d >= 760 and d <= 779 :
                new_tok[i] = 'P'
            elif d >= 780 and d <= 799 :
                new_tok[i] = 'Q'
            elif d >= 800 and d <= 999 :
                new_tok[i] = 'R'
            else:
                print d
                print "ERROR"
                exit(0);
        
    new_row = ','.join(new_tok)
    out.write(new_row)
    out.write("\n")
out.close()
