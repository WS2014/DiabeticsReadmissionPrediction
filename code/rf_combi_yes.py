import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("../data/combined/complete_cleared_pandas_train_combi_yes.csv", dtype={'admission_type_id':np.object, 'discharge_disposition_id':np.object, 'admission_source_id':np.object, 'diag_1':np.object, 'diag_2':np.object, 'diag_3':np.object})

test = pd.read_csv("../data/combined/complete_cleared_pandas_test_combi_yes.csv", dtype={'admission_type_id':np.object, 'discharge_disposition_id':np.object, 'admission_source_id':np.object, 'diag_1':np.object, 'diag_2':np.object, 'diag_3':np.object})

numeric_cols = [ 'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

#x_num_train = x_num_train.astype('int64')
#x_num_test = x_num_test.astype('int64')

"""
# scale to <0,1>
max_train = np.amax( x_num_train, 0 )
max_test = np.amax( x_num_test, 0 ) # not really needed
x_num_train = x_num_train / max_train
x_num_test = x_num_test / max_train # scale test by max_train
"""

# y
y_train = train.readmitted
y_train, mapping = pd.factorize(y_train)
y_test = test.readmitted
y_test, _ = pd.factorize(y_test)

# categorical
cat_train = train.drop( numeric_cols + [ 'readmitted'], axis = 1 )
cat_test = test.drop( numeric_cols + [ 'readmitted'], axis = 1 )

"""
x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()


# vectorize
vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )
"""
fac_x_cat_train = pd.DataFrame()
fac_x_cat_test = pd.DataFrame()
cat_cols = list(cat_train.columns.values)
for col in cat_cols:
    train_cur, _ = pd.factorize(cat_train[col])
    test_cur, _ = pd.factorize(cat_test[col])
    #x_cat_train = x_cat_train.drop(col,axis=1)
    #x_cat_test = x_cat_test.drop(col,axis=1)
    fac_x_cat_train[col] = train_cur
    fac_x_cat_test[col] = test_cur

"""  
vec_x_cat_train, _ = pd.factorize( x_cat_train )
vec_x_cat_test, _ = pd.factorize( x_cat_test )
"""

# complete x
#x_train = np.hstack(( x_num_train, vec_x_cat_train ))
#x_test = np.hstack(( x_num_test, vec_x_cat_test ))

fac_x_cat_train = fac_x_cat_train.as_matrix()
fac_x_cat_test = fac_x_cat_test.as_matrix()

x_train = np.hstack(( x_num_train, fac_x_cat_train ))
x_test = np.hstack(( x_num_test, fac_x_cat_test ))

#rf = RandomForestClassifier(n_estimators=100, max_features='auto', n_jobs=4)
rf = RandomForestClassifier(n_estimators=100, max_features=15, n_jobs=4, max_depth=8)
#rf = RandomForestClassifier(n_estimators=100, max_features='auto', n_jobs=4, max_depth=5)

#training
st = time.time()
print "training started"
rf.fit( x_train, y_train )
print "training ended"
et = time.time()
tt = et - st
print "Training Time = " + str(tt) + "\n"

#predictions
pred = rf.predict( x_test )
y_score = rf.decision_function(x_test)
out = open('../results/rf_combi_yes.txt','w')

#validation
total = y_test.size
good = 0
bad = 0
for i in range(total):
    a = y_test[i]
    p = pred[i]
    line = str(a) + ',' + str(p) + '\n'
    out.write(line)
    if str(a) == str(p):
        good = good + 1;
    else:
        bad = bad + 1;
    
accuracy = good / total
accuracy = accuracy * 100
print accuracy
out.write("\n*************\nAccuracy = " + str(accuracy) + "\n********");
out.write("\n*************\Total = " + str(total) + "\n********");
out.write(str(mapping))
out.close()

