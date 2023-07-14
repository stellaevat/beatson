import streamlit as st
import numpy as np
import pandas as pd
import math
import spacy
from tqdm import tqdm
from scipy.sparse import lil_matrix, vstack
from skmultilearn.dataset import load_from_arff
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# TODO: Replace with en_core_sci_sm
nlp = spacy.load("en_core_web_sm")

LABELLED, UNLABELLED, TEST = 0.6, 0.2, 0.2
ITERATIONS = 10
SELECTION = 10
metrics = {'Accuracy': accuracy_score,
           'Precision': precision_score,
           'Recall': recall_score,
           'F1': f1_score}
predict_msg = "Running prediction algorithm..."

@st.cache_resource(show_spinner=False)
def get_classifier():
    ## TODO: Is there any way to fix random state of OVR too?
    clf = OneVsRestClassifier(estimator=SVC(kernel='rbf', probability=True, random_state=42))
    return clf
    
@st.cache_resource(show_spinner=False)
def get_vectorizer():
    vectorizer = TfidfVectorizer(tokenizer=lambda x:[token.lemma_.lower() for token in nlp(x) if not (token.is_stop or token.is_punct or token.is_space)])
    return vectorizer


# MMC Algorithm

@st.cache_data(show_spinner=False)
def mmc_label_prediction(X_labelled, y_labelled, X_unlabelled):
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

    ## Can this be vectorised?
    for i in range(total_unlabelled):
        ln = n_y_predicted[i]
        for j in y_sorted_indices[i, :ln]:
            y_predicted[i, j] = 1

    return y_predicted

@st.cache_data(show_spinner=False)
def mmc_query_selection(X_labelled, y_labelled, X_unlabelled):
    y_decision = clf.decision_function(X_unlabelled)
    y_predicted = mmc_label_prediction(X_labelled, y_labelled, X_unlabelled)

    expected_loss_reduction_score = np.argsort(np.sum((1 - y_predicted * y_decision)/2, axis=1))[::-1]
    to_label = expected_loss_reduction_score[:SELECTION]
    return to_label
  
# Simplified MMC Algorithm 

@st.cache_data(show_spinner=False) 
def mmc_simplified_query_selection(X_unlabelled):
    y_decision = clf.decision_function(X_unlabelled)
    y_predicted = clf.predict(X_unlabelled)
    y_predicted[y_predicted < 1] = -1

    expected_loss_reduction_score = np.argsort(np.sum((1 - y_predicted * y_decision)/2, axis=1))[::-1]
    to_label = expected_loss_reduction_score[:SELECTION]
    return to_label
  
# BinMin Algorithm

@st.cache_data(show_spinner=False)
def binmin_query_selection(X_unlabelled):
    y_decision = clf.decision_function(X_unlabelled)
    most_uncertain_label_score = np.argsort(np.min(np.abs(y_decision), axis=1))
    to_label = most_uncertain_label_score[:SELECTION]
    return to_label


  
# Prediction

@st.cache_data(show_spinner=predict_msg)
def predict(X_labelled, y_labelled, X_unlaballed):
    vectorizer = get_vectorizer()
    X_labelled = vectorizer.fit_transform(X_labelled)
    X_unlabelled = vectorizer.transform(X_unlabelled)
    
    clf = get_classifier()
    clf.fit(X_labelled, y_labelled)
    
    to_label = binmin_query_selection(X_unlabelled)
    y_predicted = clf.predict(X_unlabelled)
    y_probabilities = None
    
    return y_predicted, y_probabilities, to_label
    
