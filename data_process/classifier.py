import DataPrep
import FeatureSelection
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier


#the feature selection has been done in FeatureSelection.py module. here we will create models using those features for prediction

#first we will use bag of words techniques


#building classifier using naive bayes
nb_pipeline = Pipeline([
        ('NBCV',FeatureSelection.countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(DataPrep.train_news['content'].values.astype('str'),DataPrep.train_news['credibility'])
predicted_nb = nb_pipeline.predict(DataPrep.test_news['content'].values.astype('str'))
np.mean(predicted_nb == DataPrep.test_news['credibility'])



#building classifier using logistic regression
logR_pipeline = Pipeline([
        ('LogRCV',FeatureSelection.countV),
        ('LogR_clf',LogisticRegression())
        ])

logR_pipeline.fit(DataPrep.train_news['content'].values.astype('str'),DataPrep.train_news['credibility'])
predicted_LogR = logR_pipeline.predict(DataPrep.test_news['content'].values.astype('str'))
np.mean(predicted_LogR == DataPrep.test_news['credibility'])


#building Linear SVM classfier
svm_pipeline = Pipeline([
        ('svmCV',FeatureSelection.countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(DataPrep.train_news['content'].values.astype('str'),DataPrep.train_news['credibility'])
predicted_svm = svm_pipeline.predict(DataPrep.test_news['content'].values.astype('str'))
np.mean(predicted_svm == DataPrep.test_news['credibility'])


#random forest
random_forest = Pipeline([
        ('rfCV',FeatureSelection.countV),
        ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
        ])

random_forest.fit(DataPrep.train_news['content'].values.astype('str'),DataPrep.train_news['credibility'])
predicted_rf = random_forest.predict(DataPrep.test_news['content'].values.astype('str'))
np.mean(predicted_rf == DataPrep.test_news['credibility'])


#User defined functon for K-Fold cross validatoin
def build_confusion_matrix(classifier):

    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(DataPrep.train_news):
        train_text = DataPrep.train_news.iloc[train_ind]['content'].values.astype('str')
        train_y = DataPrep.train_news.iloc[train_ind]['credibility']

        test_text = DataPrep.train_news.iloc[test_ind]['content'].values.astype('str')
        test_y = DataPrep.train_news.iloc[test_ind]['credibility']

        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)

        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)

    return (print('Total contents classified:', len(DataPrep.train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))

#K-fold cross validation for all classifiers
# build_confusion_matrix(nb_pipeline)
# build_confusion_matrix(logR_pipeline)
# build_confusion_matrix(svm_pipeline)
# build_confusion_matrix(random_forest)


"""So far we have used bag of words technique to extract the features and passed those featuers into classifiers. We have also seen the
f1 scores of these classifiers. now lets enhance these features using term frequency weights with various n-grams
"""


##Now using n-grams
#naive-bayes classifier
nb_pipeline_ngram = Pipeline([
        ('nb_tfidf',FeatureSelection.tfidf_ngram),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(DataPrep.train_news['content'].values.astype('str'),DataPrep.train_news['credibility'])
predicted_nb_ngram = nb_pipeline_ngram.predict(DataPrep.test_news['content'].values.astype('str'))
np.mean(predicted_nb_ngram == DataPrep.test_news['credibility'])


#logistic regression classifier
logR_pipeline_ngram = Pipeline([
        ('LogR_tfidf',FeatureSelection.tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logR_pipeline_ngram.fit(DataPrep.train_news['content'].values.astype('str'),DataPrep.train_news['credibility'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(DataPrep.test_news['content'].values.astype('str'))
np.mean(predicted_LogR_ngram == DataPrep.test_news['credibility'])


#linear SVM classifier
svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',FeatureSelection.tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(DataPrep.train_news['content'].values.astype('str'),DataPrep.train_news['credibility'])
predicted_svm_ngram = svm_pipeline_ngram.predict(DataPrep.test_news['content'].values.astype('str'))
np.mean(predicted_svm_ngram == DataPrep.test_news['credibility'])

#random forest classifier
random_forest_ngram = Pipeline([
        ('rf_tfidf',FeatureSelection.tfidf_ngram),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
        ])

random_forest_ngram.fit(DataPrep.train_news['content'].values.astype('str'),DataPrep.train_news['credibility'])
predicted_rf_ngram = random_forest_ngram.predict(DataPrep.test_news['content'].values.astype('str'))
np.mean(predicted_rf_ngram == DataPrep.test_news['credibility'])

#multi_layer perceptron classifier
mlp_ngram = Pipeline([
        ('mlp_tfidf',FeatureSelection.tfidf_ngram),
        ('mlp_clf',MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
        ])
mlp_ngram.fit(DataPrep.train_news['content'].values.astype('str'),DataPrep.train_news['credibility'])
predicted_mlp_ngram = mlp_ngram.predict(DataPrep.test_news['content'].values.astype('str'))
np.mean(mlp_ngram == DataPrep.test_news['credibility'])

#K-fold cross validation for all classifiers
# build_confusion_matrix(nb_pipeline_ngram)
# build_confusion_matrix(logR_pipeline_ngram)
# build_confusion_matrix(svm_pipeline_ngram)
#
# build_confusion_matrix(random_forest_ngram)
build_confusion_matrix(mlp_ngram)
