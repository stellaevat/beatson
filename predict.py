import streamlit as st
import numpy as np
from gsheets import update_sheet, clear_sheet_column, get_gsheets_urls, get_gsheets_columns, get_delimiter

project_columns, pub_columns, metric_columns = get_gsheets_columns()
UID_COL, ACC_COL, TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL, ANNOT_COL, PREDICT_COL, SCORE_COL, LEARN_COL  = project_columns
PMID_COL, PUBTITLE_COL, ABSTRACT_COL, MESH_COL, KEY_COL = pub_columns
text_columns = [TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL]

PROJECT_THRESHOLD = 10
ANNOT_SUGGESTIONS = 10
LABEL_THRESHOLD = 3

DELIMITER = get_delimiter()

@st.cache_resource(show_spinner=False)
def get_label_matrix(df):
    labels = set()
    for annotation in df[ANNOT_COL]:
        if annotation is not None:
            labels.update(set(annotation.split(DELIMITER)))
    labels = sorted(list(labels))

    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    
    y_labelled = np.zeros((len(df), len(labels)))
    for i, annotation in enumerate(df[ANNOT_COL]):
        if annotation is not None:
            for label in annotation.split(DELIMITER):
                y_labelled[i, label_to_index[label]] = 1
            
    return y_labelled, label_to_index, index_to_label
 
@st.cache_resource(show_spinner=False) 
def get_sample_matrix(df, pub_df):
    X_labelled = []
    for i, project in df.iterrows():
        text = " ".join([field for field in project[text_columns] if field is not None])
        if project[PUB_COL]:
            for pmid in project[PUB_COL].split(DELIMITER):
                text += " " + " ".join([field for field in pub_df.loc[pub_df[PMID_COL] == pmid].iloc[0] if field is not None])
        X_labelled.append(text)
    return X_labelled

@st.cache_data(show_spinner="Processing dataset...")
def process_dataset(project_df, pub_df): 
    unlabelled_df = project_df[project_df[ANNOT_COL].isnull()]
    labelled_df = project_df[project_df[ANNOT_COL].notnull()]
    y_labelled, label_to_index, index_to_label = get_label_matrix(labelled_df)
    
    label_sums = np.sum(y_labelled, axis=0)
    project_sum = np.sum(y_labelled, axis=1)
    labelled_projects = len(project_sum[project_sum > 0]) 
    rare_labels = (label_sums < LABEL_THRESHOLD).nonzero()[0]

    if labelled_projects == len(project_df):
        error = "All projects in the dataset have been manually annotated. Use the **Search** tab to find and add unannotated projects."
    elif labelled_projects < PROJECT_THRESHOLD:
        error = f"So far **{labelled_projects} projects** have been annotated. For the algorithm to work well please find at least **{PROJECT_THRESHOLD - labelled_projects}** more project{'s' if PROJECT_THRESHOLD - labelled_projects > 1 else ''} to annotate."
    elif len(rare_labels) > 0:
        error = f"Some labels have less than **{LABEL_THRESHOLD} samples**. For the algorithm to work well please find more projects to label as: **{', '.join([index_to_label[i] for i in rare_labels])}**."
    else:
        X_labelled = get_sample_matrix(labelled_df, pub_df)
        X_unlabelled = get_sample_matrix(unlabelled_df, pub_df)
        labels = sorted(list(label_to_index.keys()), key=lambda x: label_to_index[x])
        return X_labelled, X_unlabelled, y_labelled, labels, ""
        
    return None, None, None, None, error

    
@st.cache_data(show_spinner="Processing predictions...")
def process_predictions(y_predicted, y_scores, labels, df, _connection):
    labels = np.array(labels)
    project_df = st.session_state.get("project_df")
    
    to_annotate = np.argsort(y_scores)[:ANNOT_SUGGESTIONS].tolist() if len(y_scores) > ANNOT_SUGGESTIONS else np.argsort(y_scores).tolist()
    unlabelled_df = project_df[project_df[ANNOT_COL].isnull()]
    
    for i, project_id in enumerate(unlabelled_df[ACC_COL]):
        predicted_mask = np.where(y_predicted[i] > 0, True, False)
        predicted_str = DELIMITER.join(sorted(labels[predicted_mask]))
        score = "%.3f" % y_scores[i]

        old_prediction = project_df.loc[project_df[ACC_COL] == project_id, PREDICT_COL].item()
        old_prediction = old_prediction if old_prediction else ""
        old_score = project_df.loc[project_df[ACC_COL] == project_id, SCORE_COL].item()
        old_score = old_score if old_score else ""
        
        if predicted_str != old_prediction or score != old_score:
            update_sheet(_connection, project_id, {PREDICT_COL : predicted_str, SCORE_COL : score})
            project_df.loc[project_df[ACC_COL] == project_id, PREDICT_COL] = predicted_str
            project_df.loc[project_df[ACC_COL] == project_id, SCORE_COL] = score
        
    if to_annotate:
        old_learn_df = project_df[project_df[LEARN_COL].notnull()]
        new_learn_df = unlabelled_df.iloc[to_annotate, :].reset_index(drop=True)
        
        updates = {project[ACC_COL] : None for (i, project) in old_learn_df.iterrows() if project[ACC_COL] != new_learn_df[new_learn_df[LEARN_COL] == project[LEARN_COL]][ACC_COL].item()}
        for (i, project) in new_learn_df.iterrows():
            if project[ACC_COL] != old_learn_df[old_learn_df[LEARN_COL] == project[LEARN_COL]][ACC_COL].item():
                updates[project[ACC_COL]] = str(i+1)
          
        # Update with minimum API calls  
        if updates:
            if len(updates) < len(new_learn_df) + 1:
                for (project_id, order) in updates.items():
                    update_sheet(_connection, project_id, {LEARN_COL : order})
                    project_df.loc[project_df[ACC_COL] == project_id, LEARN_COL] = order
            else:
                clear_sheet_column(_connection, LEARN_COL)
                project_df[LEARN_COL] = None
                for (project_id, order) in updates.items():
                    if order is not None:
                        update_sheet(_connection, project_id, {LEARN_COL : order})
                        project_df.loc[project_df[ACC_COL] == project_id, LEARN_COL] = order
    
    
    # Clear predictions from earlier runs (labelled projects not updated above)
    labelled_df = project_df[project_df[ANNOT_COL].notnull()]
    for i, project_id in enumerate(labelled_df[ACC_COL]):
        if project_df.loc[project_df[ACC_COL] == project_id, SCORE_COL].item():
            update_sheet(connection, project_id, {PREDICT_COL : None, SCORE_COL : None})
            project_df.loc[project_df[ACC_COL] == project_id, [PREDICT_COL, SCORE_COL]] = None