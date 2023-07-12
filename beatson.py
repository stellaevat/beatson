import re
import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from Bio import Entrez
from collections import defaultdict
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
type_col = "Data Type"
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
        ".ag-cell": {"padding": "0px 12px;"}
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



def id_to_url(base_url, page_id):
    return base_url + str(page_id) + "/"
    
def id_to_html_link(base_url, page_id):
    return f'<a target="_blank" href="{base_url + str(page_id) + "/"}">{page_id}</a>'
    
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
    
def update_annotation(gsheet_url_proj, project_id, labels):
    update = f"""
            UPDATE "{gsheet_url_proj}"
            SET {label_col} = "{labels}"
            WHERE {acc_col} = "{project_id}"
            """
    connection.execute(update)
    
def submit_labels(project_id, gsheet_url_proj):
    new = st.session_state.get("new", "")
    labels = st.session_state.get("labels", "")
    
    if project_id:
        combined = (set(labels) | {l.strip() for l in new.split(",")}) - {""}
        labels_str = DELIMITER.join(sorted(list(combined)))
        
        if combined ^ set(original_labels):
            update_annotation(gsheet_url_proj, project_id, labels_str)
            if labels_str:
                active_df.at[selected_row_index, label_col] = labels_str
            else:
                active_df.at[selected_row_index, label_col] = None
            
    st.session_state.new = ""
    
@st.cache_data(show_spinner=False)
def esearch(database, terms):
    if not terms:
        return None
    handle = Entrez.esearch(db=database, term=terms, retmax=retmax, idtype="acc")
    ids = Entrez.read(handle)["IdList"]
    return ids

 
@st.cache_data(show_spinner=search_msg)
def api_search(search_terms):
    search_df, search_pub_df = None, None
    search_terms = [term.strip() for term in search_terms.split() if term.strip()]
    # TODO: find synonyms
    
    ids = esearch(project_db, "+AND+".join(search_terms))
    all_project_data, all_pub_data = retrieve_projects(ids)
    if all_project_data:
        search_df = pd.DataFrame(all_project_data)
        if all_pub_data:
            search_pub_df = pd.DataFrame(all_pub_data)
                
    return search_df, search_pub_df
            
            
@st.cache_data(show_spinner=search_msg)
def local_search(search_terms, project_df):
    search_terms = [term.strip() for term in search_terms.split() if term.strip()]
    search_expr = r"(\b(" + "|".join(search_terms) + r")\b)"
    # TODO: find synonyms
    
    raw_counts = np.column_stack([project_df.astype(str)[col].str.count(search_expr, flags=re.IGNORECASE) for col in project_df])
    total_counts = np.sum(raw_counts, axis=1)
    mask = np.where(total_counts > 0, True, False)
    search_df = project_df.loc[mask]
    
    search_df = search_df.sort_index(axis=0, key=lambda col: col.map(lambda i: total_counts[i]), ascending=False, ignore_index=True)
    return search_df
    
    
@st.cache_data(show_spinner=False)
def build_interactive_grid(active_df, starting_page, selected_row_index):
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
    
    builder = GridOptionsBuilder.from_dataframe(active_df[aggrid_columns])
    
    builder.configure_column(acc_col, lockPosition="left", suppressMovable=True, width=110)
    builder.configure_column(title_col, flex=3.5)
    builder.configure_column(label_col, flex=1)
    builder.configure_selection() # Required for interactive selection
    builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=results_per_page)
    builder.configure_grid_options(**options_dict)
    builder.configure_side_bar()

    gridOptions = builder.build()
    return gridOptions

def show_details():
    st.session_state.project_details_hidden = False
    
def hide_details():
    st.session_state.project_details_hidden = True
    
def display_project_details(project):
    st.subheader(f"{project[title_col] if project[title_col] else project[name_col] if project[name_col] else project[acc_col]}")
    
    df = pd.DataFrame(project[detail_columns])
    df.loc[acc_col] = id_to_html_link(base_project_url, project[acc_col])
    
    if project[pub_col]:
        df.loc[pub_col] = DELIMITER.join([id_to_html_link(base_pub_url, pub_id) for pub_id in project[pub_col].split(DELIMITER)])
    
    for field in detail_columns:
        if not project[field]:
            df = df.drop(field)
            
    st.write(df.to_html(render_links=True, escape=False), unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    if project[descr_col]:
        st.write(project[descr_col])

    
st.title("BioProject Annotation")

# LOCAL SEARCH FUNCTION

st.header("Search")

search = st.text_input("Search:", label_visibility="collapsed", key="search").strip()
if st.session_state.get("prev_search", "") != search:
    st.session_state.selected_row_index = 0
st.session_state.prev_search = search
st.write("")

connection = connect_gsheets_api()
project_df = load_data(gsheet_url_proj, project_columns)
pub_df = load_data(gsheet_url_pub, pub_columns)

if search:     
    search_df = local_search(search, project_df)
    if search_df is not None and not search_df.empty:
        st.write(f"Search results for '{search}':")
        active_df = search_df
    else:
        st.write(f"No search results for '{search}'. All projects:")
        active_df = project_df
else:
    st.write(f"All projects:")
    active_df = project_df

# PROJECT DATAFRAME

rerun = st.session_state.get("rerun", 0)
selected_row_index = st.session_state.get("selected_row_index", 0)
starting_page = selected_row_index // results_per_page

if not active_df.empty:
    gridOptions = build_interactive_grid(active_df, starting_page, selected_row_index)
    grid = AgGrid(
            active_df[aggrid_columns], 
            gridOptions=gridOptions,
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
    previous_page = int(st.session_state.get("starting_page", 0))
    project_details_hidden = st.session_state.get("project_details_hidden", True)
    
    if project_details_hidden:
        show_details = st.button("Show details", key="show_button", on_click=show_details)
    else:
        hide_details = st.button("Hide details", key="hide_button", on_click=hide_details)
    st.write("")
    st.write("")

    if rerun:
        if not project_details_hidden:
            display_project_details(active_df.iloc[selected_row_index])
        st.session_state.starting_page = starting_page
        st.session_state.rerun = 0
        
    elif not selected_df.empty:
        selected_mask = active_df[acc_col].isin(selected_df[acc_col])
        selected_data = active_df.loc[selected_mask]
        
        selected_row_index = selected_data.index.tolist()[0]
        st.session_state.selected_row_index = selected_row_index
        st.session_state.rerun = 1
        
        st.experimental_rerun()    
    else:
        if not project_details_hidden:
            display_project_details(active_df.iloc[selected_row_index])
            

# ANNOTATION FUNCTION

st.header("Annotate")
        
project_id = active_df.iloc[selected_row_index][acc_col]
label_options = set()
if not project_df.empty:
    for annotation in project_df[label_col]:
        if annotation is not None:
            label_options.update(set(annotation.split(DELIMITER)))
    label_options = sorted(list(label_options))
    
original_labels = project_df.at[selected_row_index, label_col]
if original_labels is None:
    original_labels = []
else:
    original_labels = original_labels.split(DELIMITER)

with st.form(key="Annotate"):
    st.write(f"Edit {project_id} labels:")
    if label_options:
        col1, col2 = st.columns(2)
        with col1:
            labels = st.multiselect("", label_options, default=original_labels, label_visibility="collapsed", key="labels")
        with col2:
            new = st.text_input("", placeholder="Or create new (comma-separated)", label_visibility="collapsed", key="new")
    else:
        labels = ""
        new = st.text_input("", placeholder="Create new (comma-separated)", label_visibility="collapsed", key="new")
        
    submit_button = st.form_submit_button("Update", on_click=submit_labels, args=(project_id, gsheet_url_proj))

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
        message.write(f"Some labels have less than {LABEL_THRESHOLD} samples. For the algorithm to work well please find more projects to label as any of the following: {', '.join([index_to_label[i] for i in rare_labels])}.")
        return None, None
    else:
        message.write("Ready to predict. You will be asked for some more annotations as the algorithm is running.")
        labelled_project_df = project_df.loc[project_df[label_col] is not None]
        X_labelled = get_sample_matrix(labelled_project_df, pub_df, text_columns, pub_col, pmid_col, DELIMITER)
        return X_labelled, y_labelled
        
        
        
st.header("Predict")
st.write("Run prediction algorithm:")
message = st.empty()
start_button = st.button("Start")
st.write("")
st.write("")
    
if start_button:
    X_labelled, y_labelled = check_dataset(message)
    if X_labelled and y_labelled:
        y_predicted, y_probabilities = predict(X_labelled, y_labelled)

    
