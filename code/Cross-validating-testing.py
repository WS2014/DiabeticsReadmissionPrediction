import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, average_precision_score
from sklearn.feature_extraction import DictVectorizer as DV
from matplotlib.backends.backend_pdf import PdfPages



#NOTE: change dataset here
all = pd.read_csv("../data/intuition/complete_cleared_all_diag_compress.csv", dtype={'admission_type_id':np.object, 'discharge_disposition_id':np.object, 'admission_source_id':np.object, 'diag_1':np.object, 'diag_2':np.object, 'diag_3':np.object})
out = open('../results/int_1.txt','w')

numeric_cols = [ 'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

x_num_all = all[ numeric_cols ].as_matrix()

# y
y_all = all.readmitted
y_all, mapping = pd.factorize(y_all)

# Binarize the output
n_classes = mapping.size
y_all = label_binarize(y_all, classes=[0, 1, 2])
n_classes = y_all.shape[1]

# categorical
cat_all = all.drop( numeric_cols + [ 'readmitted'], axis = 1 )

#NOTE: change Preprocessing to Vectorization or Fatorization dependin on classifier
"""
#Vectorize
x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )
#"""

#"""
#Factorize
fac_x_cat_all = pd.DataFrame()
cat_cols = list(cat_all.columns.values)
for col in cat_cols:
    all_cur, _ = pd.factorize(cat_all[col])
    fac_x_cat_all[col] = all_cur

fac_x_cat_all = fac_x_cat_all.as_matrix()
#"""

x_all = np.hstack(( x_num_all, fac_x_cat_all ))

#train-test split

def get_scores(nt,mh):
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=.7,random_state=24)
    total = x_all.shape[0]
    quant = total/5;
    train_sets = dict()
    for i in range(5):
        j = i + 1
        ai = i * quant
        bi = j * quant - 1
        train_sets[i] = (x_train, x_val, y_train, y_val)
    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=10, max_features='auto', n_jobs=1, max_depth=0))
    clf.fit( x_train, y_train )


    

nt = 0
mh = 0
while nt < 350:
    nt = nt + 50
    while mh < 200:
        mh = mh + 5:
        rec[i],prec[i],fpr[i]],tpr[i] = get_scores(nt,mh);