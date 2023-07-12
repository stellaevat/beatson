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
from active_learning import get_label_matrix, get_sample_matrix

st.set_page_config(page_title="BioProject Annotation")

Entrez.email = "stell.aeva@hotmail.com"
project_db = "bioproject"
base_project_url = "https://www.ncbi.nlm.nih.gov/" + project_db + "/"
base_pub_url = "https://pubmed.ncbi.nlm.nih.gov/"
retmax = 10
results_per_page = 10

gsheet_url_proj = st.secrets["private_gsheets_url_proj"]
gsheet_url_pub = st.secrets["private_gsheets_url_pub"]

DELIMITER = ", "
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
label_col = "Labels"
project_columns = [uid_col, acc_col, title_col, name_col, descr_col, type_col, scope_col, org_col, pub_col, label_col]
aggrid_columns = [acc_col, title_col, label_col]
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
        search_df.columns = project_columns[:-1]
        search_df[label_col] = None
        if all_pub_data:
            search_pub_df = pd.DataFrame(all_pub_data)
            search_pub_df.columns = pub_columns
            
        # TODO: Should these be removed or just show up with their labels?
        projects_to_drop = []
        for i, row in search_df.iterrows():
            if row[acc_col] in project_df[acc_col].unique():
                projects_to_drop.append(i)
        search_df = search_df.drop(projects_to_drop)
                
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
    search = st.text_input("", label_visibility="collapsed", placeholder="Search", key=(tab + "_search")).strip()
    st.write("")
    
    if st.session_state.get(tab + "_prev_search", "") != search:
        st.session_state[tab + "_selected_row_index"] = 0
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
            df = df.drop(field)
            
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
def get_grid_options(df, starting_page, selected_row_index):
    options_dict = {
        "enableCellTextSelection" : True,
        "onFirstDataRendered" : JsCode("""
            function onFirstDataRendered(params) {
                params.api.paginationGoToPage(""" + str(starting_page) + """);
            }
        """),
        "getRowStyle" : JsCode("""
            function(params) {
                if (params.rowIndex == """ + str(selected_row_index) + """) {
                    return {'background-color': '""" + primary_colour + """', 'color': 'black'};
                }
            }
        """)
    }
    
    builder = GridOptionsBuilder.from_dataframe(df[aggrid_columns])
    
    builder.configure_column(acc_col, lockPosition="left", suppressMovable=True, width=110)
    builder.configure_column(title_col, flex=3.5)
    builder.configure_column(label_col, flex=1)
    builder.configure_selection() # Required for interactive selection
    builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=results_per_page)
    builder.configure_grid_options(**options_dict)
    builder.configure_side_bar()

    grid_options = builder.build()
    return grid_options
    
def display_interactive_grid(tab, df):
    rerun = st.session_state.get(tab + "_rerun", 0)
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    starting_page = selected_row_index // results_per_page

    grid_options = get_grid_options(df, starting_page, selected_row_index)
    grid = AgGrid(
        df[aggrid_columns], 
        gridOptions=grid_options,
        width="100%",
        theme="alpine",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
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
        st.session_state[tab + "_rerun"] = 1
        
        st.experimental_rerun()    
    else:
        if not project_details_hidden:
            display_project_details(df.iloc[selected_row_index])


def update_annotation(project_id, labels):
    update = f"""
            UPDATE "{gsheet_url_proj}"
            SET {label_col} = "{labels}"
            WHERE {acc_col} = "{project_id}"
            """
    connection.execute(update)
    
def insert_annotation(values):
    values = ["'" + str(val) if str(val).isnumeric() else val for val in values]
    values_str = '("' + '", "'.join([str(val) for val in values]) + '")'
    insert = f''' 
            INSERT INTO "{gsheet_url_proj}" ({", ".join(project_columns)})
            VALUES {values_str}
            '''
    connection.execute(insert)
    
def insert_publication(values):
    values = ["'" + str(val) if str(val).isnumeric() else val for val in values]
    values_str = '("' + '", "'.join([str(val) for val in values]) + '")'
    insert = f''' 
            INSERT INTO "{gsheet_url_pub}" ({", ".join(pub_columns)})
            VALUES {values_str}
            '''
    connection.execute(insert)
 
def get_project_labels(project_id):
    if project_id in project_df[acc_col].unique():
        project_labels = project_df[project_df[acc_col] == project_id][label_col].values[0]
        if project_labels is not None:
            return project_labels.split(DELIMITER)
            
    return []
        
 
def update_labels(tab, df, pub_df=None):
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
                project_df.loc[project_df[acc_col] == project_id, label_col] = updated_labels_str
            else:
                project_df.loc[project_df[acc_col] == project_id, label_col] = None
    else:
        # Global variable used so display actually changed
        api_project_df.at[selected_row_index, label_col] = updated_labels_str
        project_values = df.iloc[selected_row_index]
        
        project_df.loc[len(project_df.index)] = project_values.tolist()
        insert_annotation(project_values.tolist())
        
        if pub_df is not None and project_values[pub_col] is not None:
            for pmid in project_values[pub_col].split(DELIMITER):
                if pmid in pub_df[pmid_col].unique():
                    pub_values = pub_df.loc[pub_df[pmid_col] == pmid].squeeze()
                    insert_publication(pub_values.tolist())
            
    st.session_state[tab + "_new"] = ""

def display_annotation_feature(tab, df, pub_df=None):
    st.header("Annotate")
    
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = df.iloc[selected_row_index][acc_col]
    
    # TODO: May want to move this outside method
    label_options = set()
    if not project_df.empty:
        for annotation in project_df[label_col]:
            if annotation is not None:
                label_options.update(set(annotation.split(DELIMITER)))
        label_options = sorted(list(label_options))
    
    original_labels = get_project_labels(project_id)

    with st.form(key=(tab + "_annotation_form")):
        st.write(f"Edit {project_id} labels:")
        if label_options:
            col1, col2 = st.columns(2)
            with col1:
                labels = st.multiselect("", label_options, default=original_labels, label_visibility="collapsed", key=(tab + "_labels"))
            with col2:
                new = st.text_input("", placeholder="Or create new (comma-separated)", label_visibility="collapsed", key=(tab + "_new"))
        else:
            labels = ""
            new = st.text_input("", placeholder="Create new (comma-separated)", label_visibility="collapsed", key=(tab + "_new"))
            
        st.form_submit_button("Update", on_click=update_labels, args=(tab, df, pub_df))
 
def check_dataset(message):
    y_labelled, label_to_index, index_to_label = get_label_matrix(project_df, label_col, DELIMITER)
    label_sums = np.sum(y_labelled, axis=0)
    project_sum = np.sum(y_labelled, axis=1)
    labelled_projects = len(project_sum[project_sum > 0]) 
    rare_labels = (label_sums < LABEL_THRESHOLD).nonzero()[0]

    if labelled_projects < PROJECT_THRESHOLD:
        message.write(f"So far {labelled_projects} projects have been annotated. For the algorithm to work well please find at least {PROJECT_THRESHOLD - labelled_projects} more project{'s' if PROJECT_THRESHOLD - labelled_projects > 1 else ''} to annotate.") 
        return None, None
    elif len(rare_labels) > 0:
        message.write(f"Some labels have less than {LABEL_THRESHOLD} samples. For the algorithm to work well please find more projects to label as any of the following: {', '.join([index_to_label[i] for i in rare_labels])}. Use the Search tab to search the BioProject database directly if the annotation dataset is insufficient.")
        return None, None
    else:
        message.write("Ready to predict. You will be asked for some more annotations as the algorithm is running.")
        labelled_project_df = project_df.loc[project_df[label_col] is not None]
        X_labelled = get_sample_matrix(labelled_project_df, pub_df, text_columns, pub_col, pmid_col, DELIMITER)
        return X_labelled, y_labelled
        
        
 
st.title("BioProject Annotation")
annotate_tab, search_tab, predict_tab = st.tabs(["Annotate", "Search", "Predict"])

connection = connect_gsheets_api()
project_df = load_data(gsheet_url_proj, project_columns)
pub_df = load_data(gsheet_url_pub, pub_columns)
    
with annotate_tab:
    st.header("Find projects")
    tab_1 = "Annotate"
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
            display_interactive_grid(tab_1, annotate_df)
            display_annotation_feature(tab_1, annotate_df)
        
    else:
        st.write("Annotation dataset unavailable. Use the Search tab to search the BioProject database directly.")

with search_tab:
    st.header("Search BioProject for more projects")
    tab_2 = "Search"
    
    api_terms = display_search_feature(tab_2)
    if api_terms:
        api_project_df, api_pub_df = api_search(api_terms)
        if api_project_df is not None and not api_project_df.empty:
            st.write(f"Results for '{api_terms}':")
            display_interactive_grid(tab_2, api_project_df)
            display_annotation_feature(tab_2, api_project_df, api_pub_df)
        else:
            st.write(f"No results for '{api_terms}'. Check for typos or try looking for something else.")
        
with predict_tab:
    st.header("Predict annotations")
    message = st.empty()
    start_button = st.button("Start")
    st.write("")
    st.write("")
        
    # if start_button:
        # X_labelled, y_labelled = check_dataset(message)
        # if X_labelled and y_labelled:
            # X_unlabelled, to_label, not_to_label = learn(X_labelled, y_labelled)
            
            # if to_label:
                # learn_df = pd.DataFrame(X_unlabelled[to_label])
        # new_labels = None
        
        # y_labelled = np.vstack((y_labelled, new_labels))
        # X_labelled = vstack((X_labelled, X_unlabelled[to_label]))
        # X_unlabelled = X_unlabelled[not_to_label]

        # clf.fit(X_labelled, y_labelled)
        
        # st.session_state.iteration = iteration + 1
