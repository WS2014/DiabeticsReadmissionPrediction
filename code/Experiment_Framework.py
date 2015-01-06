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

#NOTE: change dataset here
all = pd.read_csv("../data/complete_cleared_pandas.csv", dtype={'admission_type_id':np.object, 'discharge_disposition_id':np.object, 'admission_source_id':np.object, 'diag_1':np.object, 'diag_2':np.object, 'diag_3':np.object})

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
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=.7,random_state=24)

#NOTE: change classifier here
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_features=15, n_jobs=4, max_depth=5))

#training
st = time.time()
print "training started"
clf.fit( x_train, y_train )
print "training ended"
et = time.time()
tt = et - st
print "Training Time = " + str(tt) + "\n"

#predictions
pred = clf.predict( x_test )
#NOTE: change to decision_function or predict_proba depending on the classifier
y_score = clf.predict_proba(x_test)
#y_score = clf.decision_function(x_test)
out = open('../results/rf_ALL_ova.txt','w')

#################################################################################
#PrecisionRecall-plot
precision = dict()
recall = dict()
PR_area = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], thresholds = precision_recall_curve(y_test[:,i],y_score[:,i])
    PR_area[i] = auc(recall[i], precision[i])
    average_precision[i] = average_precision_score(y_test[:,i], y_score[:, i])

# Compute micro-average ROC curve and ROC PR_area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score, average='micro')
# Plot Precision-Recall curve for each class
plt.clf()
plt.plot(recall["micro"], precision["micro"], label='micro-average Precision-recall curve (PR_area = {0:0.2f})'.format(average_precision["micro"]))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], label='Precision-recall curve of class {' + mapping[i] + '}' + ' (PR_area = {1:0.2f})'.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()
#############################################################################
#ROC-plot
# Compute ROC curve and ROC ROC_area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC ROC_area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (ROC_area = {0:0.2f})'.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {' + mapping[i] + '}' + ' (ROC_area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

###########################################
#Confusion matrix
confusion_mat =(y_test, y_pred, mapping)
print confusion_mat

###########################################
for i in range(n_classes):
    print "Precision-Recall*AUC: {" + mapping[i] + "} => " + str(PR_area[i]) + "\n"
    print "ReceiverOperatingCharacheristics*AUC: {" + mapping[i] + "} => " + str(ROC_area[i]) + "\n"

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
out.write("\n*************\Total = " + str(total) + "\n********\n");
out.write(str(confusion_mat))
out.write("\n")
out.write(str(mapping))
out.write("\n*******Area Below Curve*********\n")
for i in range(n_classes):
    out.write("Precision-Recall*AUC: {" + mapping[i] + "} => " + str(PR_area[i]) + "\n")
    out.write("ReceiverOperatingCharacheristics*AUC: {" + mapping[i] + "} => " + str(ROC_area[i]) + "\n")
out.write("\n**********\nTraining Time = " + str(tt) + "\n*************\n")
out.close()

