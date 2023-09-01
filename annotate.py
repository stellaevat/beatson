import streamlit as st
import pandas as pd
from gsheets import update_sheet, insert_sheet, get_gsheets_urls, get_gsheets_columns, get_delimiter

GSHEETS_URL_PROJ, GSHEETS_URL_PUB, GSHEETS_URL_METRICS = get_gsheets_urls()

project_columns, pub_columns, metric_columns = get_gsheets_columns()

UID_COL, ACC_COL, TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL, ANNOT_COL, PREDICT_COL, SCORE_COL, LEARN_COL  = project_columns
PMID_COL, PUBTITLE_COL, ABSTRACT_COL, MESH_COL, KEY_COL = pub_columns

DELIMITER = get_delimiter()

def add_to_dataset(tab, connection, add_df, add_pub_df):
    project_df = st.session_state.get("project_df", pd.DataFrame(columns=project_columns))
    pub_df = st.session_state.get("pub_df", pd.DataFrame(columns=pub_columns))
    
    for i, project in pd.DataFrame(add_df).iterrows():
        if project[ACC_COL] not in project_df[ACC_COL].unique():
            project_df.loc[len(project_df.index)] = project.tolist()
            insert_sheet(connection, project.fillna(value='').tolist())
            
            publications = project[PUB_COL]
            if publications and add_pub_df is not None:
                for pmid in publications.split(DELIMITER):
                    if pmid in add_pub_df[PMID_COL].unique() and pmid not in pub_df[PMID_COL].unique():
                        pub_values = add_pub_df.loc[add_pub_df[PMID_COL] == pmid].squeeze()
                        insert_sheet(connection, pub_values.tolist(), pub_columns, GSHEETS_URL_PUB)
                        pub_df.loc[len(pub_df)] = pub_values
                        
            # Display update
            add_df.drop(i, axis=0, inplace=True)
            selected_projects = st.session_state.get(tab + "_selected_projects", [])
            if project[ACC_COL] in selected_projects:
                selected_projects.remove(project[ACC_COL])
                st.session_state[tab + "_selected_projects"] = selected_projects
                
    st.session_state[tab + "_selected_row_index"] = 0
                        
                    
def display_add_to_dataset_feature(tab, connection, add_df, add_pub_df):
    st.header("Add to dataset")
    col1, col2, col3 = st.columns(3)
    
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = add_df.iloc[selected_row_index][ACC_COL]
    selected_projects = st.session_state.get(tab + "_selected_projects", [])
    
    with st.form(key=(tab + "_add_form")):
        add_selection = st.multiselect("Add selection", add_df[ACC_COL], default=selected_projects, label_visibility="collapsed", key=(tab + "_add_selection"))
        
        # Buttons as close as possible without line-breaks
        col1, col2 = st.columns([0.818, 0.192])
        with col1:
            st.form_submit_button("Add selection", on_click=add_to_dataset, args=(tab, connection, add_df[add_df[ACC_COL].isin(add_selection)], add_pub_df))
        with col2:
            st.form_submit_button("Add all results", on_click=add_to_dataset, args=(tab, connection, add_df, add_pub_df))

def get_project_labels(project_id, column=ANNOT_COL):
    project_df = st.session_state.get("project_df", pd.DataFrame(columns=project_columns))
    if project_id in project_df[ACC_COL].unique():
        project_labels = project_df[project_df[ACC_COL] == project_id][column].item()
        if project_labels:
            return project_labels.split(DELIMITER)
    return []
    

def update_labels(tab, connection, df, new_pub_df=None):
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = df.iloc[selected_row_index][ACC_COL]
    
    existing = st.session_state.get(tab + "_labels", "")
    new = st.session_state.get(tab + "_new", "")
    updated_labels = (set(existing) | {n.strip() for n in new.split(",")}) - {""}
    updated_labels_str = DELIMITER.join(sorted(list(updated_labels)))
    
    project_df = st.session_state.get("project_df", pd.DataFrame(columns=project_columns))
    if project_id in project_df[ACC_COL].unique():
        original_labels = get_project_labels(project_id)
        if updated_labels ^ set(original_labels):
            update_sheet(connection, project_id, {ANNOT_COL : updated_labels_str})
            
            if updated_labels_str:
                project_df.loc[project_df[ACC_COL] == project_id, ANNOT_COL] = updated_labels_str
            else:
                project_df.loc[project_df[ACC_COL] == project_id, ANNOT_COL] = None
    else:
        add_to_dataset(df.iloc[selected_row_index], new_pub_df)
            
    st.session_state[tab + "_new"] = ""
    

@st.cache_data(show_spinner=False)    
def get_label_options(project_df):
    label_options = set()
    if not project_df.empty:
        for annotation in project_df[ANNOT_COL]:
            if annotation is not None:
                label_options.update(set(annotation.split(DELIMITER)))
        label_options = sorted(list(label_options))
    return label_options
    
    
def display_annotation_feature(tab, connection, df, new_pub_df=None, allow_new=True):
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = df.iloc[selected_row_index][ACC_COL]
    
    project_df = st.session_state.get("project_df", pd.DataFrame(columns=project_columns))
    label_options = get_label_options(project_df)
    original_labels = get_project_labels(project_id)
    st.session_state[tab + "_labels"] = original_labels

    with st.form(key=(tab + "_annotation_form")):
        st.write(f"Edit **{project_id}** annotation:")
        if label_options and allow_new:
            col1, col2 = st.columns(2)
            with col1:
                labels = st.multiselect("Choose", label_options, label_visibility="collapsed", key=(tab + "_labels"))
            with col2:
                new = st.text_input("Create new", placeholder="Or create new (comma-separated)", label_visibility="collapsed", autocomplete="off", key=(tab + "_new"))
        elif label_options:
            labels = st.multiselect("Choose", label_options, label_visibility="collapsed", key=(tab + "_labels"))
        else:
            labels = ""
            new = st.text_input("Create new", placeholder="Create new (comma-separated)", label_visibility="collapsed", autocomplete="off", key=(tab + "_new"))

        st.form_submit_button("Update", on_click=update_labels, args=(tab, connection, df, new_pub_df))