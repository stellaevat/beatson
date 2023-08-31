import streamlit as st
import numpy as np
import spacy
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

nlp = spacy.load("en_core_sci_sm")

@st.cache_resource(show_spinner=False)
def get_classifier(probability=True):
    clf = OneVsRestClassifier(estimator=SVC(kernel='rbf', probability=probability, random_state=42))
    return clf
    
@st.cache_resource(show_spinner=False)
def get_vectorizer():
    vectorizer = TfidfVectorizer(tokenizer=lambda x:[token.lemma_.lower() for token in nlp(x) if not (token.is_stop or token.is_punct or token.is_space)])
    return vectorizer
    

# MMC Algorithm

@st.cache_data(show_spinner=False)
def mmc_label_prediction(clf, X_labelled, X_unlabelled, y_labelled):
    # Training data for label-number (n) prediction
    y_probability = clf.predict_proba(X_labelled)
    y_normalized = (y_probability.T / np.sum(y_probability, axis=1)).T
    n_X_train = np.sort(y_normalized, axis=1)
    n_y_train = np.sum(y_labelled, axis=1)

    # Data to predict for label-number (n) prediction
    y_probability = clf.predict_proba(X_unlabelled)
    y_normalized = (y_probability.T / np.sum(y_probability, axis=1)).T
    n_X_predict = np.sort(y_normalized, axis=1)

    # Indices of labels from most to least probable
    y_sorted_indices = np.argsort(y_normalized, axis=1)[:, ::-1]

    # Label-number (n) prediction
    label_clf = LogisticRegression(random_state=42)
    label_clf.fit(n_X_train, n_y_train)
    n_y_predicted = label_clf.predict(n_X_predict)

    # Keep n most probable labels for each sample
    total_labels = y_labelled.shape[1]
    total_unlabelled = X_unlabelled.shape[0]
    y_predicted = np.full((total_unlabelled, total_labels), -1)

    for i in range(total_unlabelled):
        ln = n_y_predicted[i]
        for j in y_sorted_indices[i, :ln]:
            y_predicted[i, j] = 1

    return y_predicted

@st.cache_data(show_spinner=False)
def mmc_scoring(clf, X_labelled, X_unlabelled, y_labelled, y_probabilities):
    y_decision = clf.decision_function(X_unlabelled)
    y_predicted = mmc_label_prediction(clf, X_labelled, X_unlabelled, y_labelled)

    expected_loss_reduction_score = np.mean((1 - y_predicted * y_decision)/2, axis=1)[::-1]
    return expected_loss_reduction_score
  
  
# Simplified MMC Algorithm 

def mmc_simple_scoring(clf, X_labelled, X_unlabelled, y_labelled, y_probabilities):
    y_decision = clf.decision_function(X_unlabelled)
    y_predicted = clf.predict(X_unlabelled)
    y_predicted[y_predicted < 1] = -1

    expected_loss_reduction_score = np.mean((1 - y_predicted * y_decision)/2, axis=1)[::-1]
    return expected_loss_reduction_score
  
  
# BinMin Algorithm

def binmin_scoring(clf, X_labelled, X_unlabelled, y_labelled, y_probabilities):
    y_decision = clf.decision_function(X_unlabelled)
    most_uncertain_label_score = np.min(np.abs(y_decision), axis=1)
    return most_uncertain_label_score
  
  
# Modified algorithms to use already obtained probabilites

def mmc_simple_proba_scoring(clf, X_labelled, X_unlabelled, y_labelled, y_probabilities):
    y_decision = y_probabilities * 2 - 1
    y_predicted = np.where(y_probabilities >= 0.5, 1, -1)

    expected_loss_reduction_score = np.mean((1 - y_predicted * y_decision)/2, axis=1)[::-1]
    return expected_loss_reduction_score   
    
def binmin_proba_scoring(clf, X_labelled, X_unlabelled, y_labelled, y_probabilities):
    most_uncertain_label_score = np.min(np.abs(y_probabilities-0.5), axis=1)
    return most_uncertain_label_score

  
# Prediction

algorithms = {
    "mmc" : mmc_scoring,
    "mmc_simple" : mmc_simple_scoring,
    "binmin" : binmin_scoring,
    "mmc_simple_proba" : mmc_simple_proba_scoring,
    "binmin_proba" : binmin_proba_scoring,
}

@st.cache_data(show_spinner="Running prediction algorithm...")
def get_predictions(X_labelled, X_unlabelled, y_labelled, alg="mmc_simple_proba"):
    vectorizer = get_vectorizer()
    X_labelled = vectorizer.fit_transform(X_labelled)
    X_unlabelled = vectorizer.transform(X_unlabelled)
    
    clf = get_classifier()
    clf_val = get_classifier(probability=False)
    
    folds = 5
    scoring = ["f1_micro", "f1_macro"]
    scores = cross_validate(clf_val, X_labelled, y_labelled, scoring=scoring, cv=folds, error_score=0)
    
    f1_micros = scores["test_f1_micro"]
    f1_macros = scores["test_f1_macro"]
    
    f1_micro_ci = stats.t.interval(confidence=0.95, df=(folds-1), loc=np.mean(f1_micros), scale=stats.sem(f1_micros))
    f1_macro_ci = stats.t.interval(confidence=0.95, df=(folds-1), loc=np.mean(f1_macros), scale=stats.sem(f1_macros))
    
    clf.fit(X_labelled, y_labelled)
    
    y_probabilities = clf.predict_proba(X_unlabelled)
    y_scores = algorithms[alg](clf, X_labelled, X_unlabelled, y_labelled, y_probabilities)
    y_predicted = np.where(y_probabilities >= 0.5, 1, 0)

    return y_predicted, y_scores, f1_micro_ci, f1_macro_ci