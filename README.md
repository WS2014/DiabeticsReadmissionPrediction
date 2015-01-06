DiabeticsReadmissionPrediction
==============================

Research Project at NITK CMU Winter School 2014: Diabetes Readmission Rate Analysis for efficient hospital resource management and increasing eficiency of initial diagnosis.

<h1>Analysis of Hospital Readmission Rates</h1>
<br></br>
<h3>A.	Introduction</h3>
<h6>1.	IMPORTANCE AND THE ISSUE!</h6>
<p>In India and many other countries, we can observe patients waiting in long queue for a doctor and many patients dying in the emergency wards with no doctor present at the right time. Moreover, many patients loose lives due to lack of medical equipment, nurses, lab tests and ambulances. This calls for an efficient hospital resource management.</p>
<p>Many diseases like cancer, heart diseases can be cured if they are diagnosed at initial stages. In majority of cases, patients get admitted due to some illness and doctors might miss the actual complication at the first diagnoses due to which the patient develop severe complications later, which might lead to his death. Hence the efficiency of initial diagnosis needs to be strengthened.</p>
<p>On the other hand, dealing with readmissions of the same patient would consume resources for a given population in a timeframe, increasing the cost for overall medical treatment. Hence reducing cost associated with readmission is very important to any nation.</p>

<h6>2.	WHAT ARE WE DOING? THE PROBLEM!</h6>
<p>Keenly observing the above issues, we can identify that they are closely related to readmissions in the hospitals. Hence by predicting if a patient gets readmitted within a month we can approximate the readmission rates which helps to suggest better resource management in terms of type and number of doctors, medical equipment etc. needed at a specific time and place. Further by identifying the hidden traits in the patients (targeting each disease independently) due to which they are getting readmitted, we can infer some additional conditions which can be verified against in the initial stages of diagnoses thereby anticipating the occurrence of future complications hence increasing the efficiency of diagnosis.</p>
<p>We tried to build an artificially intelligent model which can classify those patients who might get readmitted within a month by analyzing data pertaining to medical history of a patient. The model also finds statistically significant patterns in the features of patients which can determine his fast hospital readmission. In turn we can identify the importance of each feature of medical documentation with respect to its ability to predict readmissions. We have chosen to address the readmission rates of diabetic patients as diabetes is one of the most prevalent diseases in most of the countries.</p>

<h6>3.	HOW CAN OUR MODEL ADDRESS TO THE SPECIFIED ISSUE? THE IMPACT!</h6>
<p>Predicting Hospital Readmission Rates can increase the efficiency of initial treatment at hospitals which can save lot of lives. Once a patient gets discharged from hospital, he would get readmitted within a month, only if the initial treatment was unsuccessful or if a complication occurs which the doctor would not have anticipated at the first diagnosis. During the readmission the complications would be more severe and the patient might even die. Since our model can predict whether a person will get readmitted in 30 days, doctor can delay the initial discharge to inspect other possible complications (say cancer, heart diseases etc.) at the initial stages itself, which can prevent the occurrences of severe complications in the future and hence save lives. </p>
<p>This can reduce the number of hospital readmissions, reducing the cost associated with readmissions which can span to billions of dollars (ref). It can be instrumental in efficient hospital resource management (number and type of doctors needed, ambulances, nurses, medical equipment etc.). In turn this helps in crowd management, which is of utmost important in highly populated places. As the number of inpatients of a particular disease can be predicted, the hospital can be prepared with appropriate number of doctors of particular specialization and necessary equipment.</p>
<p>Mining hidden patterns in the diagnosis, medications, lab test results and basic characteristics of patient related to readmissions, our model finds strong set of statistically significant implications or association rules. A ranked list of such rules can be instrumental for a doctor prior to diagnosis. As an additional safety check, doctor can verify the prevalence of these conditions to every patient, increasing the effectiveness of diagnosis and better medical decisions.</p>

<h3>B.	Problem Statement</h3>
<p>To Be Updated</p>

<h3>C.	Methodology</h3>
<div>
<i><b>Preprocessing</b></i>: Elimination/imputation of feature values. 
<i><b>Feature Selection</b></i>: Identifying similar features, ignoring redundant and invariant features. 
<i><b>Classification</b></i>:
<ul>
<li><i>Lone Classifiers</i>: Decision trees, Naïve Bayes, Bayesian Networks.</li>
<li><i>Ensemble models</i>: Random Forests, AdaBoost with Decision Trees, Bagging.</li>
<li><i>Neural Networks</i></li>
</ul>
<i><b>Validation</b></i>: Plotting ROCs, tabulating AUCs.
</div>
<div>
<i><b>Association Rule Mining</b></i>:Apriori Algorithm to mine hidden patterns in the data.
<p>The hidden patterns contained within the data were mined using Apriori algorithm. Initially general rules were obtained which contain the patterns existing within the features. Following which Class association rules were mined, which contains patterns in the features which contribute towards a particular class of readmission.</p>
<p>A list of prominent class association rules ranked by their support value will be instrumental in representing the hidden patterns among diabetic patients.</p>
</div>

<p><i><b>NOTE</b>: Implemetation is done in python Scikit Learn, Matlab and WEKA. Scikit-Learn implementations are available in this repository (The file Experiment_ALl_Plot30.py is the latest code that is used to generate the result uploaded in the result folder). Respective results are in ALL_PLOT_30.pdf. WEKA implementation can be found online, or can be used in WEKA GUI. The results from WEKA are available in the consolidated folder.</i></p>

<h3>D.	Results and Discussions</h3>
<div>
<p>Experimentation was done with various machine learning algorithms for classifying the patients into three classes namely those who are getting readmitted within 30 days, those who are getting readmitted after 30 days, or those who are not readmitted by learning various features of patients.</p>
<p>We can observe that the obtained from several algorithms has very low recall on less than 30 day readmission. This is due to the non-uniform distribution in the dataset which is greatly biased towards NO readmissions and greater than 30 day readmissions. Hence, an additional preprocessing step was taken. Training set of 75 thousand instances was separated from remaining about 23 thousand test instances. Then instances of less than 30 day readmissions in training data was duplicated 5 times and then randomized. Once the algorithm gets trained on such data, all the following results are based on the predictions from algorithms on the original testing data.</p>
<p>Neural Networks forward feeding algorithm (MLP Classifier in WEKA) with one hidden layer with 2 hidden nodes stands out by giving highest area under Receiver Operating Characteristics curve i.e. 0.676 for the class - readmission less than 30 days, making it the best suited algorithm for the data. It also provides higher recall on less than 30 day readmissions which is a favorable case in hospital scenarios. Misclassification of greater than readmission is justified as it is an ambiguous class which can wither mean 31 day readmission or readmission after 9 years. If instances with greater than 30 day readmissions are classified either as less than 30 or NO readmission can be considered legit. In addition, the classification of NO readmission instances as greater than 30 can be considered legitimate.</p>
<p>Apriori algorithm retrieved several general and class association rules. An attempt made to predict readmissions by considering these association rules as feature values did not result in promising results, hence that approach was discontinued. But these association rules by themselves are a treasure which would help doctors for improving efficiency of initial diagnosis. Prominent Association Rules have been given in the detailed report. More of these rules have been put up in the appendix 4.3. These rules can be interpreted by inference logic from the Consequent (LHS) to Antecedent (RHS). For example, the rule (Age=[70-80] AND diag_1=786.0 AND diag_3=401.0) implies that the patients of age group 70-80 who are primarily diagnosed with conditions originating in perinatal period and whose tertiary diagnosis was diseases from circulatory disorders implies that there are higher chances of readmission after 30 days. Interpreting these rules help us discover very critical knowledge which would be trenchant in the diagnosis of patients.</p>
</div>

<h3>E.	Conclusion and Future Work</h3>
<div>
<ul>
<li>Area under ROC of 0.676 is better than previous studies (performed with lesser data).</li> 
<li>Higher recall is obtained for readmission within 30 days, which is a conservative approach appropriate for a medical domain.</li>
<li>Providing actual time of readmission lends more resolution, rather than categorizing readmissions as greater than 30 or NO. Currently, a person coming on the 31st day also belongs to the same category as a person coming after 10 years.</li>
<li>A lot of features were found to not contribute to the prediction; hence hospitals could look into more information about the patients. Features like season, and time of admission might affect the readmission rate.</li>
</ul>
</div>

<h3>F.	References</h3>
<div>
<pre>
[D1] Data Set Information: List of features and their descriptions in the initial dataset, Available at: http://www.hindawi.com/journals/bmri/2014/781670/tab1/, Visited on: 5th January, 2015
[L1] Silverstein, Marc D., et al. "Risk factors for 30-day hospital readmission in patients≥ 65 years of age." Proceedings (Baylor University. Medical Center) 21.4 (2008): 363.
[L2] Harrison, Patricia L., et al. "The impact of postdischarge telephonic follow-up on hospital readmissions." Population health management 14.1 (2011): 27-32.
[L3] Dungan, Kathleen M. "The effect of diabetes on hospital readmissions." Journal of diabetes science and technology 6.5 (2012): 1045-1052.
[L4] Eby, Elizabeth, et al. "Predictors of 30-day hospital readmission in patients with type 2 diabetes: A retrospective, case-control, database study." Current medical research and opinion 0 (2014): 1-27.
[L5] Soysal, Ömer M. "Association rule mining with mostly associated sequential patterns." Expert Systems with Applications (2014).
[A1] Yuan, M. & Lin, Y. On the non-negative garrotte estimator Journal of the Royal Statistical Society (Series B), 2007, 69, 143-161
[A2] Hawarah, Lamis, Ana Simonet, and Michel Simonet. "Dealing with missing values in a probabilistic decision tree during classification." Mining Complex Data. Springer Berlin Heidelberg, 2009. 55-74.
[A3] Chow-Liu Tree, Available at: http://en.wikipedia.org/wiki/Chow%E2%80%93Liu_tree, Visited on: 5th January, 2015
[A4] List of ICD-9 Codes, Available at: http://en.wikipedia.org/wiki/List_of_ICD-9_codes, Visited on: 5th January, 2015
[A5] MLP Classifier, Available at: http://grepcode.com/file/repo1.maven.org/maven2/nz.ac.waikato.cms.weka/multiLayerPerceptrons/1.0.1/weka/classifiers/functions/MLPClassifier.java, Visited on: 5th January, 2015
[A6] Broyden–Fletcher–Goldfarb–Shanno algorithm , Available at: http://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm, Visited on: 5th January, 2015
[A7] Apriori Algorithm, Available at: http://en.wikipedia.org/wiki/Apriori_algorithm ,Visited on: 5th January, 2015
</pre>
</div>
