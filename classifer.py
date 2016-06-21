# see http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# import packages

import os
import pandas as pd
import numpy as np

# home directory
os.chdir("/home/ryanceros/Dropbox_UNCC/Dropbox/UNCC_SocialScience_Collaboration")

# load dataset
name = 'updated.csv'
dataset = pd.read_csv(name)

# combine Title and Journal Name into one "text" field
dataset["text"] = dataset["Title"] + " " + dataset["Source.title"] + " " + dataset["Authors"] + " " + dataset["Year"].map(str)

# partition to label (train) and non-label or missing college (test)

train = dataset[(dataset['University'] == 'UNCC') & (dataset['Department'] != 'Other')]           
test = dataset[(dataset['University'] == 'UNCC') & (dataset['Department'] == 'Other')]  

# partition labelled into 80% train and 20% validation

tmp = pd.DataFrame(np.random.randn(train.shape[0], 2))
msk = np.random.rand(len(tmp)) < 0.8
valid = train[~msk]
train = train[msk]

# run tokenization
from sklearn.feature_extraction.text import CountVectorizer
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(train['Title'])
#X_train_counts.shape

# run TFIDF weight functions
from sklearn.feature_extraction.text import TfidfTransformer
#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#X_train_tfidf.shape

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB().fit(X_train_tfidf, train['College'])

# pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])
                     
# train and predict on training dataset                     
text_clf = text_clf.fit(train['text'], train['Department'])
pred_nb_train = text_clf.predict(train['text'])
np.mean(pred_nb_train == train['Department'])  
# Department: 0.73721606200882761

# score 

pred_nb_valid = text_clf.predict(valid['text'])
np.mean(pred_nb_valid == valid['Department'])  
# Department: 0.67125721901377167

pred_nb_test = text_clf.predict(test['text'])

# SVM

from sklearn.linear_model import SGDClassifier
text_svm_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                         alpha=1e-3, n_iter=5, random_state=42)),])
                                         
text_svm_clf = text_svm_clf.fit(train['text'], train['Department'])
pred_svm_train = text_svm_clf.predict(train['text'])
np.mean(pred_svm_train == train['Department']) 
# Department: 0.89740553342663365

pred_svm_valid = text_svm_clf.predict(valid['text'])
np.mean(pred_svm_valid == valid['Department']) 
# Department: 0.81919147045757446

pred_svm_test = text_clf.predict(test['text'])


# tuning parameters
from sklearn.grid_search import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),}
              
gs_clf = GridSearchCV(text_svm_clf, parameters, n_jobs=-1)

gs_clf = gs_clf.fit(train['text'], train['Department'])

best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
    
# optimized SVM
text_svm_clf = Pipeline([('vect', CountVectorizer(ngram_range = (1,1))),
                    ('tfidf', TfidfTransformer(use_idf='True')),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                         alpha=1e-3, n_iter=5, random_state=42)),])
                                         
text_svm_clf = text_svm_clf.fit(train['text'], train['Department'])
pred_svm_train = text_svm_clf.predict(train['text'])
np.mean(pred_svm_train == train['Department']) 
# Department: 0.89740553342663365

pred_svm_valid = text_svm_clf.predict(valid['text'])
np.mean(pred_svm_valid == valid['Department']) 
# Department: 0.81919147045757446

xtab = pd.crosstab(pred_svm_valid,valid['Department'])
xtab.to_csv('conf_matrix_dept.csv')

pred_svm_test = text_clf.predict(test['text'])

groups = pd.DataFrame(train.groupby(['Department','College']).groups.keys())
groups.columns = ['Department','College_Rep']


# export test authors to CSV
predictions = pd.DataFrame(pred_svm_test)
predictions.columns = ['Pred_Dept']

predictions = predictions.set_index(test.index)

test['Department'] = predictions

test = test.merge(groups, how='inner', on='Department')

test['College'] = test['College_Rep']
test.drop('College_Rep', 1)


test.to_csv('missing_depts.csv')
