import streamlit as st
import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
from Bio import Entrez
from collections import defaultdict
from streamlit.components.v1 import html
from shillelagh.backends.apsw.db import connect
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from dataset import retrieve_projects
from active_learning import predict

st.set_page_config(page_title="BioProject Annotation")

tab_names = ["Annotate", "Search", "Predict"]
tab_1, tab_2, tab_3 = tab_names

Entrez.email = "stell.aeva@hotmail.com"
project_db = "bioproject"
base_project_url = "https://www.ncbi.nlm.nih.gov/" + project_db + "/"
base_pub_url = "https://pubmed.ncbi.nlm.nih.gov/"
retmax = 10
results_per_page = 10

gsheet_url_proj = st.secrets["private_gsheets_url_proj"]
gsheet_url_pub = st.secrets["private_gsheets_url_pub"]

DELIMITER = ", "
EMPTY_VALUE = "-"
PROJECT_THRESHOLD = 10
LABEL_THRESHOLD = 3

uid_col = "UID"
acc_col = "Accession"
title_col = "Title"
name_col = "Name"
descr_col = "Description"
type_col = "Data_Type"
scope_col = "Scope"
org_col = "Organism"
pub_col = "PMIDs"
annot_col = "Annotation"
predict_col = "Prediction"
learn_col = "To_Annotate"
project_columns = [uid_col, acc_col, title_col, name_col, descr_col, type_col, scope_col, org_col, pub_col, annot_col, predict_col, learn_col]
aggrid_columns = [acc_col, title_col, annot_col]
aggrid_prediction_columns = [acc_col, title_col, predict_col]
detail_columns = [acc_col, type_col, scope_col, org_col, pub_col]
text_columns = [title_col, name_col, descr_col, type_col, scope_col, org_col]

pmid_col = "PMID"
pubtitle_col = "Title"
abstract_col = "Abstract"
mesh_col = "MeSH"
key_col = "Keywords"
pub_columns = [pmid_col, pubtitle_col, abstract_col, mesh_col, key_col]

search_msg = "Getting search results..."
loading_msg = "Loading project data..."
checking_msg = "Checking dataset..."

primary_colour = "#81b1cc"
aggrid_css = {
        "#gridToolBar": {"display": "none;"},
        ".ag-theme-alpine, .ag-theme-alpine-dark": {"--ag-font-size": "12px;"},
        ".ag-cell": {"padding": "0px 12px;"},
    }
streamlit_css = r'''
    <style>
        h3 {font-size: 1.5rem; color: ''' + primary_colour + ''';}
        thead {display : none;}
        th {color: ''' + primary_colour + ''';}
        [data-testid="stForm"] {border: 0px; padding: 0px;}
        [kind="secondaryFormSubmit"] {position: absolute; right: 0px;}
        [kind="secondary"] {position: absolute; right: 0px;}
    </style>
'''
st.markdown(streamlit_css, unsafe_allow_html=True)
    
@st.cache_resource(show_spinner=loading_msg)
def connect_gsheets_api():
    connection = connect(
        ":memory:",
        adapter_kwargs = {
            "gsheetsapi": { 
                "service_account_info":  dict(st.secrets["gcp_service_account"])
            }
        }
    )
    return connection
    
@st.cache_resource(show_spinner=loading_msg)
def load_data(gsheet_url, _columns):
    query = f'SELECT * FROM "{gsheet_url}"'
    executed_query = connection.execute(query)
    df = pd.DataFrame(executed_query.fetchall())
    if not df.empty:
        df.columns = _columns
        # df = df.fillna(value=EMPTY_VALUE)
    return df
    
@st.cache_data(show_spinner=False)
def esearch(database, terms):
    if not terms:
        return None
    handle = Entrez.esearch(db=database, term=terms, retmax=retmax, idtype="acc")
    ids = Entrez.read(handle)["IdList"]
    return ids

 
@st.cache_resource(show_spinner=search_msg)
def api_search(search_terms):
    search_df, search_pub_df = None, None
    search_terms = [term.strip().lower() for term in search_terms.split() if term.strip()]
    # TODO: find synonyms
    
    ids = esearch(project_db, "+OR+".join(search_terms))
    all_project_data, all_pub_data = retrieve_projects(ids)
    if all_project_data:
        search_df = pd.DataFrame(all_project_data)
        search_df.columns = project_columns[:-3]
        search_df[annot_col] = None
        if all_pub_data:
            search_pub_df = pd.DataFrame(all_pub_data)
            search_pub_df.columns = pub_columns
            
        for i, row in search_df.iterrows():
            project_id = row[acc_col]
            if project_id in project_df[acc_col].unique():
                search_df.at[i, annot_col] = project_df[project_df[acc_col] == project_id][annot_col].item()
                
    return search_df, search_pub_df
            
            
@st.cache_resource(show_spinner=search_msg)
def local_search(search_terms, df):
    search_terms = [term.strip() for term in search_terms.split() if term.strip()]
    search_expr = r"(\b(" + "|".join(search_terms) + r")\b)"
    # TODO: find synonyms
    
    raw_counts = np.column_stack([df.astype(str)[col].str.count(search_expr, flags=re.IGNORECASE) for col in text_columns])
    total_counts = np.sum(raw_counts, axis=1)
    mask = np.where(total_counts > 0, True, False)
    search_df = df.loc[mask]
    
    search_df = search_df.sort_index(axis=0, key=lambda col: col.map(lambda i: total_counts[i]), ascending=False, ignore_index=True)
    return search_df
    
def display_search_feature(tab):
    search = st.text_input("Search", label_visibility="collapsed", placeholder="Search", key=(tab + "_search")).strip()
    st.write("")
    
    if st.session_state.get(tab + "_prev_search", "") != search:
        st.session_state[tab + "_selected_row_index"] = 0
        st.session_state[tab + "_selected_projects"] = []
    st.session_state[tab + "_prev_search"] = search
    
    return search

def id_to_url(base_url, page_id):
    return f'<a target="_blank" href="{base_url + str(page_id) + "/"}">{page_id}</a>'
   
def display_project_details(project):
    st.write("")
    st.subheader(f"{project[title_col] if project[title_col] else project[name_col] if project[name_col] else project[acc_col]}")
    
    df = pd.DataFrame(project[detail_columns])
    df.loc[acc_col] = id_to_url(base_project_url, project[acc_col])
    
    if project[pub_col]:
        df.loc[pub_col] = DELIMITER.join([id_to_url(base_pub_url, pub_id) for pub_id in project[pub_col].split(DELIMITER)])
    
    for field in detail_columns:
        if not project[field]:
            df = df.drop(field, axis=0)
            
    st.write(df.to_html(render_links=True, escape=False), unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    if project[descr_col]:
        st.write(project[descr_col])
        
def show_details(tab):
    st.session_state[tab + "_project_details_hidden"] = False
    
def hide_details(tab):
    st.session_state[tab + "_project_details_hidden"] = True

@st.cache_data(show_spinner=False)
def get_grid_options(df, columns, starting_page, selected_row_index, selection_mode="single"):
    options_dict = {
        "enableCellTextSelection" : True,
        "onFirstDataRendered" : JsCode("""
            function onFirstDataRendered(params) {
                params.api.paginationGoToPage(""" + str(starting_page) + """);
            }
        """),
    }
    
    if selection_mode == "single":
        options_dict["getRowStyle"] = JsCode("""
            function(params) {
                if (params.rowIndex == """ + str(selected_row_index) + """) {
                    return {'background-color': '""" + primary_colour + """', 'color': 'black'};
                }
            }
        """)
    
    builder = GridOptionsBuilder.from_dataframe(df[columns])
    
    builder.configure_column(acc_col, lockPosition="left", suppressMovable=True, width=110)
    builder.configure_column(title_col, flex=3)
    builder.configure_column(columns[-1], flex=1)
    builder.configure_selection(selection_mode=selection_mode)
    builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=results_per_page)
    builder.configure_grid_options(**options_dict)
    builder.configure_side_bar()

    grid_options = builder.build()
    return grid_options
    
def display_interactive_grid(tab, df, columns, selection_mode="single"):
    rerun = st.session_state.get(tab + "_rerun", 0)
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    starting_page = selected_row_index // results_per_page

    grid_options = get_grid_options(df, columns, starting_page, selected_row_index, selection_mode)
    grid = AgGrid(
        df[columns].fillna(value=EMPTY_VALUE), 
        gridOptions=grid_options,
        width="100%",
        theme="alpine",
        update_mode=(GridUpdateMode.SELECTION_CHANGED if selection_mode=="single" else GridUpdateMode.NO_UPDATE),
        custom_css=aggrid_css,
        allow_unsafe_jscode=True,
        reload_data=False,
        enable_enterprise_modules=False
    )
    selected_row = grid['selected_rows']
    selected_df = pd.DataFrame(selected_row)
    previous_page = st.session_state.get(tab + "_starting_page", 0)
    project_details_hidden = st.session_state.get(tab + "_project_details_hidden", True)
    
    if project_details_hidden:
        st.button("Show details", key=(tab + "_show"), on_click=show_details, args=(tab,))
    else:
        st.button("Hide details", key=(tab + "_hide"), on_click=hide_details, args=(tab,))
    st.write("")

    if rerun:
        if not project_details_hidden:
            display_project_details(df.iloc[selected_row_index])
        st.session_state[tab + "_starting_page"] = starting_page
        st.session_state[tab + "_rerun"] = 0
        
    elif not selected_df.empty:
        selected_mask = df[acc_col].isin(selected_df[acc_col])
        selected_data = df.loc[selected_mask]
        
        selected_row_index = selected_data.index.tolist()[0]
        st.session_state[tab + "_selected_row_index"] = selected_row_index
        st.session_state[tab + "_selected_projects"] = st.session_state.get(tab + "_selected_projects", []) + [df.iloc[selected_row_index][acc_col]]
        st.session_state[tab + "_rerun"] = 1
        
        st.experimental_rerun()    
    else:
        if not project_details_hidden:
            display_project_details(df.iloc[selected_row_index])


def update_annotation(project_id, annotation):
    update = f"""
            UPDATE "{gsheet_url_proj}"
            SET {annot_col} = "{annotation}"
            WHERE {acc_col} = "{project_id}"
            """
    connection.execute(update)
    
def update_to_annotate(project_id, to_annotate):
    update = f"""
            UPDATE "{gsheet_url_proj}"
            SET {learn_col} = "{to_annotate}"
            WHERE {acc_col} = "{project_id}"
            """
    connection.execute(update)
    
def update_predicted(project_id, predicted):
    update = f"""
            UPDATE "{gsheet_url_proj}"
            SET {predict_col} = "{predicted}"
            WHERE {acc_col} = "{project_id}"
            """
    connection.execute(update)
    
def insert_annotation(values):
    values_str = '("' + '", "'.join([str(val) for val in values]) + '")'
    insert = f''' 
            INSERT INTO "{gsheet_url_proj}" ({", ".join(project_columns)})
            VALUES {values_str}
            '''
    connection.execute(insert)
    
def insert_publication(values):
    values_str = '("' + '", "'.join([str(val) for val in values]) + '")'
    insert = f''' 
            INSERT INTO "{gsheet_url_pub}" ({", ".join(pub_columns)})
            VALUES {values_str}
            '''
    connection.execute(insert)
 
def get_project_labels(project_id):
    if project_id in project_df[acc_col].unique():
        project_labels = project_df[project_df[acc_col] == project_id][annot_col].item()
        if project_labels:
            return project_labels.split(DELIMITER)
    return []
 
def update_labels(tab, df, new_pub_df=None):
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = df.iloc[selected_row_index][acc_col]
    
    existing = st.session_state.get(tab + "_labels", "")
    new = st.session_state.get(tab + "_new", "")
    updated_labels = (set(existing) | {n.strip() for n in new.split(",")}) - {""}
    updated_labels_str = DELIMITER.join(sorted(list(updated_labels)))
    
    if project_id in project_df[acc_col].unique():
        original_labels = get_project_labels(project_id)
        if updated_labels ^ set(original_labels):
            update_annotation(project_id, updated_labels_str)
            
            if updated_labels_str:
                project_df.loc[project_df[acc_col] == project_id, annot_col] = updated_labels_str
                if project_df.loc[project_df[acc_col] == project_id, learn_col].item() == True:
                    st.session_state[tab_3 + "_selected_row_index"] = 0
            else:
                project_df.loc[project_df[acc_col] == project_id, annot_col] = None
    else:
        # Global variable used so that display is actually changed
        api_project_df.at[selected_row_index, annot_col] = updated_labels_str
        
        project_values = df.iloc[selected_row_index].tolist()
        project_values += ['' for i in range(len(project_columns) - len(project_values))]
        
        project_df.loc[len(project_df.index)] = project_values
        insert_annotation(project_values)
        
        publications = df.at[selected_row_index, pub_col]
        if publications and new_pub_df is not None:
            for pmid in publications.split(DELIMITER):
                if pmid in new_pub_df[pmid_col].unique() and pmid not in pub_df[pmid_col].unique():
                    pub_values = new_pub_df.loc[new_pub_df[pmid_col] == pmid].squeeze()
                    insert_publication(pub_values.tolist())
                    pub_df.loc[len(pub_df)] = pub_values
            
    st.session_state[tab + "_new"] = ""

@st.cache_data(show_spinner=False)    
def get_label_options(project_df):
    label_options = set()
    if not project_df.empty:
        for annotation in project_df[annot_col]:
            if annotation is not None:
                label_options.update(set(annotation.split(DELIMITER)))
        label_options = sorted(list(label_options))
    return label_options

def display_annotation_feature(tab, df, new_pub_df=None, allow_new=True):
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = df.iloc[selected_row_index][acc_col]
    
    label_options = get_label_options(project_df)
    original_labels = get_project_labels(project_id)
    st.session_state[tab + "_labels"] = original_labels

    with st.form(key=(tab + "_annotation_form")):
        st.write(f"Edit **{project_id}** labels:")
        if label_options and allow_new:
            col1, col2 = st.columns(2)
            with col1:
                labels = st.multiselect("Choose", label_options, label_visibility="collapsed", key=(tab + "_labels"))
            with col2:
                new = st.text_input("Create new", placeholder="Or create new (comma-separated)", label_visibility="collapsed", key=(tab + "_new"))
        elif label_options:
            labels = st.multiselect("Choose", label_options, label_visibility="collapsed", key=(tab + "_labels"))
        else:
            labels = ""
            new = st.text_input("Create new", placeholder="Create new (comma-separated)", label_visibility="collapsed", key=(tab + "_new"))

        st.form_submit_button("Update", on_click=update_labels, args=(tab, df, new_pub_df)) 
        

def display_add_to_dataset_feature(tab, df):
    st.header("Add to dataset")
    col1, col2, col3 = st.columns(3)
    
    # with col1:
        # add_selection = st.button("Add selection")
    # with col2:
        # st.write("or")
    # with col3:
        # add_all = st.button("Add all")
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = df.iloc[selected_row_index][acc_col]
    selected_projects = st.session_state.get(tab + "_selected_projects", [])
    
    with st.form(key=(tab + "_add_form")):
        add_selection = st.multiselect("Add selection:", df[acc_col], default=selected_projects, key=(tab + "_add_selection"))
        
        st.form_submit_button("Add", args=(add_selection,))
        
        
@st.cache_resource(show_spinner=checking_msg)
def get_label_matrix(df):
    labels = set()
    for annotation in df[annot_col]:
        if annotation is not None:
            labels.update(set(annotation.split(DELIMITER)))
    
    i = 0
    label_to_index = {}
    index_to_label = {}
    for label in labels:
        label_to_index[label] = i
        index_to_label[i] = label
        i += 1
    
    y_labelled = np.zeros((len(df), len(labels)))
    for i, annotation in enumerate(df[annot_col]):
        if annotation is not None:
            for label in annotation.split(DELIMITER):
                y_labelled[i, label_to_index[label]] = 1
            
    return y_labelled, label_to_index, index_to_label
 
@st.cache_resource(show_spinner=checking_msg) 
def get_sample_matrix(df, pub_df):
    X_labelled = []
    for i, project in df.iterrows():
        text = " ".join([field for field in project[text_columns] if field is not None])
        if project[pub_col]:
            for pmid in project[pub_col].split(DELIMITER):
                text += " " + " ".join([field for field in pub_df.loc[pub_df[pmid_col] == pmid].iloc[0] if field is not None])
        X_labelled.append(text)
    return X_labelled
 
@st.cache_data(show_spinner="Checking dataset...") 
def check_dataset(project_df):
    unlabelled_df = project_df[project_df[annot_col].isnull()]
    labelled_df = project_df[project_df[annot_col].notnull()]
    y_labelled, label_to_index, index_to_label = get_label_matrix(labelled_df)
    
    label_sums = np.sum(y_labelled, axis=0)
    project_sum = np.sum(y_labelled, axis=1)
    labelled_projects = len(project_sum[project_sum > 0]) 
    rare_labels = (label_sums < LABEL_THRESHOLD).nonzero()[0]

    if labelled_projects == len(project_df):
        message.write("All projects in the dataset have been manually annotated. Use the **Search** tab to find and add unannotated projects.")
    elif labelled_projects < PROJECT_THRESHOLD:
        message.write(f"So far **{labelled_projects} projects** have been annotated. For the algorithm to work well please find at least **{PROJECT_THRESHOLD - labelled_projects}** more project{'s' if PROJECT_THRESHOLD - labelled_projects > 1 else ''} to annotate.") 
    elif len(rare_labels) > 0:
        message.write(f"Some labels have less than **{LABEL_THRESHOLD} samples**. For the algorithm to work well please find more projects to label as: **{', '.join([index_to_label[i] for i in rare_labels])}**.")
    else:
        X_labelled = get_sample_matrix(labelled_df, pub_df)
        X_unlabelled = get_sample_matrix(unlabelled_df, pub_df)
        labels = sorted(list(label_to_index.keys()), key=lambda x: label_to_index[x])
        return X_labelled, X_unlabelled, y_labelled, labels
        
    return None, None, None, None

@st.cache_data(show_spinner=False) 
def int_column(col):
    return pd.Series([int(val) if (val and val.isnumeric()) else 0 for val in col])
    
@st.cache_data(show_spinner="Processing predictions...")
def process_predictions(y_predicted, y_probabilities, to_annotate, labels, df):
    labels = np.array(labels)
    unlabelled_df = project_df[project_df[annot_col].isnull()]
    
    for i, project_id in enumerate(unlabelled_df[acc_col]):
        predicted_mask = np.where(y_predicted[i] > 0, True, False)
        predicted_str = DELIMITER.join(labels[predicted_mask])
        # TODO: Takes way too long, make async?
        # update_predicted(project_id, predicted_str)
        project_df.loc[project_df[acc_col] == project_id, predict_col] = predicted_str
        
    if to_annotate:
        old_learn_df = project_df[int_column(project_df[learn_col]) > 0]
        for project_id in old_learn_df[acc_col]:
            update_to_annotate(project_id, "0")
            project_df.loc[project_df[acc_col] == project_id, learn_col] = "0"
        
        learn_df = unlabelled_df.iloc[to_annotate, :]
        for i, project_id in enumerate(learn_df[acc_col]):
            update_to_annotate(project_id, str(i+1))
            project_df.loc[project_df[acc_col] == project_id, learn_col] = str(i+1)
 
st.title("BioProject Annotation")
annotate_tab, search_tab, predict_tab = st.tabs(tab_names)

connection = connect_gsheets_api()
project_df = load_data(gsheet_url_proj, project_columns)
pub_df = load_data(gsheet_url_pub, pub_columns)
    
with annotate_tab:
    st.header("Annotate projects")
    annotate_df = project_df

    if not project_df.empty:
        search_terms = display_search_feature(tab_1)
        
        if search_terms:   
            search_df = local_search(search_terms, project_df)
            if search_df is not None and not search_df.empty:
                st.write(f"Results for '{search_terms}':")
                annotate_df = search_df
            else:
                st.write(f"No results for '{search_terms}'. All projects:")
        
        if not annotate_df.empty:
            display_interactive_grid(tab_1, annotate_df, aggrid_columns)
            display_annotation_feature(tab_1, annotate_df)
        
    else:
        st.write("Annotation dataset unavailable. Use the Search tab to search the BioProject database directly.")
    

with search_tab:
    st.header("Search BioProject")
    
    api_terms = display_search_feature(tab_2)
    if api_terms:
        api_project_df, api_pub_df = api_search(api_terms)
        if api_project_df is not None and not api_project_df.empty:
            st.write(f"Results for '{api_terms}':")
            display_interactive_grid(tab_2, api_project_df, aggrid_columns)
            # display_add_to_dataset_feature(tab_2, api_project_df)
            display_annotation_feature(tab_2, api_project_df, api_pub_df)
        else:
            st.write(f"No results for '{api_terms}'. Check for typos or try looking for something else.")
  
with predict_tab:
    st.header("Predict annotations")
    
    message = st.empty()
    message.write("Click **Start** to get label predictions for all unannotated projects.")
    start_button = st.button("Start", key="start_button")
    st.write("")
    st.write("")

    if start_button:
        X_labelled, X_unlabelled, y_labelled, labels = check_dataset(project_df)
        if X_labelled:
            y_predicted, y_probabilities, to_annotate = predict(X_labelled, y_labelled, X_unlabelled)
            
            # Columns irrelevant to method cacheing dropped
            df = project_df.drop([predict_col, learn_col], axis=1)
            process_predictions(y_predicted, y_probabilities, to_annotate, labels, df)
            
            st.session_state.new_predictions = True
    
    predict_df = project_df[project_df[predict_col].notnull()]
    if predict_df is not None and not predict_df.empty:
        if st.session_state.get("new_predictions", False):
            st.header("Predicted labels")
        else:
            st.header("Previously predicted labels")
        display_interactive_grid(tab_3, predict_df, aggrid_prediction_columns)
    
    learn_df = project_df[int_column(project_df[learn_col]) > 0]    
    if learn_df is not None and not learn_df.empty:
        st.header("Improve predictions")
        st.write("To improve performance, consider annotating the following projects:")
        # Sort by annotation importance to active learning
        learn_df = learn_df.sort_values(learn_col, axis=0, ignore_index=True, key=lambda col: int_column(col))
        display_interactive_grid("Improve", learn_df, aggrid_columns)
        display_annotation_feature("Improve", learn_df)