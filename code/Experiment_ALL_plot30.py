import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, average_precision_score
from sklearn.feature_extraction import DictVectorizer as DV
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

def eer(fpr,tpr):
    dist=1000
    mi=0
    y1=1-fpr
    for j in range(0,(fpr.size-1)):
        di=abs(tpr[j]-y1[j])
        if(di < dist):
            dist=di
            mi=j
    return y1,mi
#NOTE: change dataset and result files here
all = pd.read_csv("../data/intuition/complete_cleared_all_diag_compress_full_select.csv", dtype={'admission_type_id':np.object, 'discharge_disposition_id':np.object, 'admission_source_id':np.object})

out = open('../results/All_plot30_final.txt','w')
pp = PdfPages('../results/All_plot30_final.pdf')

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
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=.25,random_state=24)

#*Printing the distribution
print "###DATA DISTRIBUTION START###"
print "\n***ALL***"
y_all_t = y_all.T
all_tot = y_all.shape[0]
print "TOTAL: " + str(all_tot)
for i in range(n_classes):
    lab = mapping[i]
    lc = np.sum(y_all_t[i])
    print lab + ": " + str(lc)

print "\n***Training Set***"
y_train_t = y_train.T
train_tot = y_train.shape[0]
print "TOTAL: " + str(train_tot)
for i in range(n_classes):
    lab = mapping[i]
    lc = np.sum(y_train_t[i])
    print lab + ": " + str(lc)

print "\n***Testing Set***"
y_test_t = y_test.T
test_tot = y_test.shape[0]
print "TOTAL: " + str(test_tot)
for i in range(n_classes):
    lab = mapping[i]
    lc = np.sum(y_test_t[i])
    print lab + ": " + str(lc)
print "###DATA DISTRIBUTION END###"
#*

#NOTE: change classifier here
clf1 = OneVsRestClassifier(RandomForestClassifier(n_estimators=250, max_features='auto', n_jobs=4, max_depth=5))
clf2 = OneVsRestClassifier(AdaBoostClassifier(n_estimators=250, algorithm='SAMME'))
clf3 = OneVsRestClassifier(GaussianNB())
clf4 = OneVsRestClassifier(DecisionTreeClassifier())
#clf5 = OneVsRestClassifier(svm.SVC(gamma=2))

#training
st = time.time()
print "training started"
clf1.fit( x_train, y_train )
clf2.fit( x_train, y_train )
clf3.fit( x_train, y_train )
clf4.fit( x_train, y_train )
print "training ended"
et = time.time()
tt = et - st
print "Training Time = " + str(tt) + "\n"

#predictions
pred1 = clf1.predict( x_test )
pred2 = clf2.predict( x_test )
pred3 = clf3.predict( x_test )
pred4 = clf4.predict( x_test )
pred = pred2;
#NOTE: change to decision_function or predict_proba depending on the classifier
y_score1 = clf1.predict_proba(x_test)
y_score2 = clf2.predict_proba(x_test)
y_score3 = clf3.predict_proba(x_test)
y_score4 = clf4.predict_proba(x_test)
#y_score = clf.decision_function(x_test)
y_score = y_score1 + y_score2 + y_score3


#################################################################################
#PrecisionRecall-plot
precision = dict()
recall = dict()
PR_area = dict()
PR_thresholds = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], PR_thresholds[i] = precision_recall_curve(y_test[:,i],y_score[:,i])
    PR_area[i] = auc(recall[i], precision[i])
    average_precision[i] = average_precision_score(y_test[:,i], y_score[:, i])

# Compute micro-average ROC curve and ROC PR_area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score, average='micro')
# Plot Precision-Recall curve for each class
plt.clf()
#micro average plot
#plt.plot(recall["micro"], precision["micro"], label='micro-average Precision-recall curve (PR_area = {0:0.2f})'.format(average_precision["micro"]))
for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(recall[i], precision[i], label='ALL Probability Summation - Precision-recall curve of class {' + mapping[i] + '}' + ' (PR_area = {1:0.2f})'.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
###########
#1
precision = dict()
recall = dict()
PR_area = dict()
PR_thresholds = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], PR_thresholds[i] = precision_recall_curve(y_test[:,i],y_score1[:,i])
    PR_area[i] = auc(recall[i], precision[i])
    average_precision[i] = average_precision_score(y_test[:,i], y_score1[:, i])

for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(recall[i], precision[i], label='RandomForest - Precision-recall curve of class {' + mapping[i] + '}' + ' (PR_area = {1:0.2f})'.format(i, average_precision[i]))
#
#2
precision = dict()
recall = dict()
PR_area = dict()
PR_thresholds = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], PR_thresholds[i] = precision_recall_curve(y_test[:,i],y_score2[:,i])
    PR_area[i] = auc(recall[i], precision[i])
    average_precision[i] = average_precision_score(y_test[:,i], y_score2[:, i])

for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(recall[i], precision[i], label='AdaBoost - Precision-recall curve of class {' + mapping[i] + '}' + ' (PR_area = {1:0.2f})'.format(i, average_precision[i]))
#
#3
precision = dict()
recall = dict()
PR_area = dict()
PR_thresholds = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], PR_thresholds[i] = precision_recall_curve(y_test[:,i],y_score3[:,i])
    PR_area[i] = auc(recall[i], precision[i])
    average_precision[i] = average_precision_score(y_test[:,i], y_score3[:, i])

for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(recall[i], precision[i], label='NaiveBayes - Precision-recall curve of class {' + mapping[i] + '}' + ' (PR_area = {1:0.2f})'.format(i, average_precision[i]))
#
"""#4
precision = dict()
recall = dict()
PR_area = dict()
PR_thresholds = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], PR_thresholds[i] = precision_recall_curve(y_test[:,i],y_score4[:,i])
    PR_area[i] = auc(recall[i], precision[i])
    average_precision[i] = average_precision_score(y_test[:,i], y_score4[:, i])

for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(recall[i], precision[i], label='Precision-recall curve of class {' + mapping[i] + '}' + ' (PR_area = {1:0.2f})'.format(i, average_precision[i]))
#"""
###########
plt.legend(loc="upper right", prop={'size':8})
#plt.show()
plt.savefig(pp, format='pdf')
#############################################################################
#ROC-plot
#for each classifier
#######
plt.figure()
#1
fpr = dict()
tpr = dict()
ROC_thresholds = dict()
ROC_area = dict()
for i in range(n_classes):
    fpr[i], tpr[i], ROC_thresholds[i] = roc_curve(y_test[:, i], y_score1[:, i])
    ROC_area[i] = auc(fpr[i], tpr[i])

for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(fpr[i], tpr[i], label='RandomForest - ROC curve of class {' + mapping[i] + '}' + ' (ROC_area = {1:0.2f})'.format(i, ROC_area[i]))
        [y1,mi1]=eer(fpr[i],tpr[i])    
        plt.plot(fpr[i][mi1],y1[mi1],'o')
        print "EER - R => "+ mapping[i] + " =>" + str(fpr[i][mi1]) + "#"

#2
fpr = dict()
tpr = dict()
ROC_thresholds = dict()
ROC_area = dict()
for i in range(n_classes):
    fpr[i], tpr[i], ROC_thresholds[i] = roc_curve(y_test[:, i], y_score2[:, i])
    ROC_area[i] = auc(fpr[i], tpr[i])

for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(fpr[i], tpr[i], label='AdaBoost - ROC curve of class {' + mapping[i] + '}' + ' (ROC_area = {1:0.2f})'.format(i, ROC_area[i]))
        [y1,mi1]=eer(fpr[i],tpr[i])    
        plt.plot(fpr[i][mi1],y1[mi1],'o')
        print "EER - A => "+ mapping[i] + " =>" + str(fpr[i][mi1]) + "#"

#3
fpr = dict()
tpr = dict()
ROC_thresholds = dict()
ROC_area = dict()
for i in range(n_classes):
    fpr[i], tpr[i], ROC_thresholds[i] = roc_curve(y_test[:, i], y_score3[:, i])
    ROC_area[i] = auc(fpr[i], tpr[i])

for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(fpr[i], tpr[i], label='NaiveBayes - ROC curve of class {' + mapping[i] + '}' + ' (ROC_area = {1:0.2f})'.format(i, ROC_area[i]))
        [y1,mi1]=eer(fpr[i],tpr[i])    
        plt.plot(fpr[i][mi1],y1[mi1],'o')
        print "EER - N => "+ mapping[i] + " =>" + str(fpr[i][mi1]) + "#"

"""#4
fpr = dict()
tpr = dict()
ROC_thresholds = dict()
ROC_area = dict()
for i in range(n_classes):
    fpr[i], tpr[i], ROC_thresholds[i] = roc_curve(y_test[:, i], y_score4[:, i])
    ROC_area[i] = auc(fpr[i], tpr[i])

for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(fpr[i], tpr[i], label='DT - ROC curve of class {' + mapping[i] + '}' + ' (ROC_area = {1:0.2f})'.format(i, ROC_area[i]))
        [y1,mi1]=eer(fpr[i],tpr[i])    
        plt.plot(fpr[i][mi1],y1[mi1],'o')
        print "EER - D => "+ mapping[i] + " =>" + str(fpr[i][mi1]) + "#"


#"""
#######
# Compute ROC curve and ROC ROC_area for each class
fpr = dict()
tpr = dict()
ROC_thresholds = dict()
ROC_area = dict()
for i in range(n_classes):
    fpr[i], tpr[i], ROC_thresholds[i] = roc_curve(y_test[:, i], y_score[:, i])
    ROC_area[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC ROC_area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
ROC_area["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
[y1,mi1]=eer(fpr[1],tpr[1])

#plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (ROC_area = {0:0.2f})'.format(ROC_area["micro"]))
plt.plot(fpr[1],y1)
for i in range(n_classes):
    if mapping[i] == '<30':
        plt.plot(fpr[i], tpr[i], label='ALL Probability Summation - ROC curve of class {' + mapping[i] + '}' + ' (ROC_area = {1:0.2f})'.format(i, ROC_area[i]))
        [y1,mi1]=eer(fpr[i],tpr[i])    
        plt.plot(fpr[i][mi1],y1[mi1],'o')
        print "EER => "+ mapping[i] + " =>" + str(fpr[i][mi1]) + "#"

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right", prop={'size':8})

#plt.show()
plt.savefig(pp, format='pdf')
"""
###########################################
#EER - Equal Error Rate
eer = dict()
for i in range(n_classes):
    class_label = mapping[i]
    temp_threshs = ROC_thresholds[i]
    false_pos = fpr[i]
    true_pos = tpr[i]
    min_diff = 999
    thresh_at_min_diff = 999
    j_at_min = -1
    fp_at_min = -1
    
    for j in range(temp_threshs.size):
        t = temp_threshs[j]
        fp = false_pos[j]
        tp = true_pos[j]
        diff = abs(fp - tp)
        if diff < min_diff:
            min_diff = diff
            thresh_at_min_diff = t
            j_at_min = j
    
    eer[i] = (thresh_at_min_diff, j_at_min)

print "\n##########Equal-Error-Rate###########\n"
for i in range(n_classes):
    thresh_at_min_diff, j_at_min = eer[i]
    x = fpr[i][j_at_min]
    y = tpr[i][j_at_min]
    plt.plot(x,y,'x')
    print str(mapping[i]) + " => " +  str(thresh_at_min_diff) + "\n"

#plt.show()
plt.savefig(pp, format='pdf')
#pp.close()
###########################################
#"""
#Confusion matrix
for i in range(n_classes):
    print "Confusion_Matrix => {" + mapping[i] + "} :"
    confusion = confusion_matrix(y_test[:,i],pred[:,i])
    print confusion
    print "################"

###########################################
for i in range(n_classes):
    print "Precision-Recall*AUC: {" + mapping[i] + "} => " + str(PR_area[i]) + "\n"
    print "ReceiverOperatingCharacheristics*AUC: {" + mapping[i] + "} => " + str(ROC_area[i]) + "\n"

#validation
total = y_test.size
out.write("\n*************\Total = " + str(total) + "\n********\n");
out.write("\n")
out.write(str(mapping))
out.write("\n*******Area Below Curve*********\n")
for i in range(n_classes):
    out.write("Precision-Recall*AUC: {" + mapping[i] + "} => " + str(PR_area[i]) + "\n")
    out.write("ReceiverOperatingCharacheristics*AUC: {" + mapping[i] + "} => " + str(ROC_area[i]) + "\n")

#Confusion matrix
for i in range(n_classes):
    a = "Confusion_Matrix => {" + mapping[i] + "} : \n"
    confusion = confusion_matrix(y_test[:,i],pred[:,i])
    out.write(a)
    out.write(str(confusion))
    out.write("\n\n")

out.write("\n**********\nTraining Time = " + str(tt) + "\n*************\n")
out.close()
pp.close()
