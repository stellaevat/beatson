import re
import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from Bio import Entrez
from collections import defaultdict
from shillelagh.backends.apsw.db import connect
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode


st.set_page_config(page_title="BioProject Annotation")

Entrez.email = "stell.aeva@hotmail.com"
database = "bioproject"
base_url = "https://www.ncbi.nlm.nih.gov/bioproject/"
max_search_results = 10
results_per_page = 10

gsheet_url_proj = st.secrets["private_gsheets_url_proj"]
gsheet_url_pub = st.secrets["private_gsheets_url_pub"]

uid_col = "UID"
acc_col = "Accession"
title_col = "Title"
name_col = "Name"
descr_col = "Description"
type_col = "Data Type"
scope_col = "Scope"
org_col = "Organism"
pub_col = "Publications"
annot_col = "Labels"
link_col = "Link"
project_columns = [uid_col, acc_col, title_col, name_col, descr_col, type_col, scope_col, org_col, pub_col, annot_col]
aggrid_columns = [acc_col, title_col, annot_col]
detail_fields = [type_col, scope_col, org_col, link_col]

search_msg = "Getting search results..."
loading_msg = "Loading project data..."
delimiter = ", "

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



def id_to_url(project_id):
    return base_url + str(project_id) + "/"
    
def id_to_html_link(project_id):
    return f'<a target="_blank" href="{base_url + str(project_id) + "/"}">{project_id}</a>'
    
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
            SET {annot_col} = "{labels}"
            WHERE {acc_col} = "{project_id}"
            """
    connection.execute(update)
    
def submit_labels(project_id, gsheet_url_proj):
    new = st.session_state.get("new", "")
    labels = st.session_state.get("labels", "")
    
    if project_id:
        combined = (set(labels) | {l.strip() for l in new.split(",")}) - {""}
        labels_str = delimiter.join(sorted(list(combined)))
        
        if combined ^ set(original_labels):
            update_annotation(gsheet_url_proj, project_id, labels_str)
            if labels_str:
                active_df.at[selected_row_index, annot_col] = labels_str
            else:
                active_df.at[selected_row_index, annot_col] = None
            
    st.session_state.new = ""
    
@st.cache_data(show_spinner=False)
def esearch(db=database, term="", retmax=max_search_results, idtype="acc"):
    if not term:
        return None
    handle = Entrez.esearch(db=db, term=term, retmax=retmax, idtype=idtype)
    ids = Entrez.read(handle)["IdList"]
    return ids
    
@st.cache_data(show_spinner=False)
def efetch(db=database, ids="", rettype="xml", retmode="xml"):
    if not ids:
        return None
    handle = Entrez.efetch(db=db, id=ids, rettype=rettype, retmode=retmode)
    tree = ET.parse(handle)
    tag, project_dict = parse_xml(tree.getroot())
    return project_dict

def parse_xml(element):
    data_dict = dict(element.attrib)
    children = map(parse_xml, element)
    children_nodes = defaultdict(list)
    clean_nodes = {}
    for node, data in children:
        children_nodes[node].append(data)
    for node, data_list in children_nodes.items():
        clean_nodes[node] = data_list[0] if len(data_list) == 1 else data_list

    if clean_nodes:
        data_dict.update(clean_nodes)

    if element.text is not None and not element.text.isspace():
        data_dict['text'] = element.text
    if len(data_dict) == 1 and 'text' in data_dict:
        data_dict = data_dict['text']
    tag = element.tag
    return tag, data_dict
 
# TODO: Adapt to new parsing 
@st.cache_data(show_spinner=search_msg)
def api_search(search_terms):
    search_terms = [term.strip() for term in search_terms.split() if term.strip()]
    # TODO: find synonyms
    ids = esearch(term="+AND+".join(search_terms))
    if ids:
        project_dict = efetch(ids=",".join(ids))
        projects = project_dict['DocumentSummary']
        if not isinstance(projects, list):
            projects = [projects,]
        uids, titles, urls = [], [], []
        for project in projects:
            if "error" not in project:
                uids.append(project["acc"])
                titles.append(project["Project"]["ProjectDescr"]["Title"])
                urls.append(id_to_url(project["acc"]))
        if uids:
            search_df = pd.DataFrame(
                {acc_col:uids, project_col:titles, url_col:urls}, 
                columns=(acc_col, project_col, url_col)
            )
            search_df.set_index(acc_col, inplace=True)
            return search_df
            
@st.cache_data(show_spinner=search_msg)
def local_search(search_terms, project_df, pub_df):
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
    builder.configure_column(annot_col, flex=1)
    builder.configure_selection() # Required for interactive selection
    builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=results_per_page)
    builder.configure_grid_options(**options_dict)
    builder.configure_side_bar()

    gridOptions = builder.build()
    return gridOptions

def show_details():
    st.session_state.project_details_hidden = False
    return
    
def hide_details():
    st.session_state.project_details_hidden = True
    return
    
def display_project_details(project):
    st.write(f"Accession: {project[acc_col]}, ID: {project[uid_col]}")
    st.subheader(f"{project[title_col] if project[title_col] else project[name_col] if project[name_col] else project[acc_col]}")
    if project[descr_col]:
        st.write(project[descr_col])
    
    project[link_col] = id_to_html_link(project[acc_col])
    df = pd.DataFrame(project[detail_fields])
    st.write(df.to_html(render_links=True, escape=False), unsafe_allow_html=True)

    
st.title("BioProject Annotation")

# SEARCH FUNCTION

st.header("Search")

search = st.text_input("Search:", label_visibility="collapsed", key="search").strip()
if st.session_state.get("prev_search", "") != search:
    st.session_state.selected_row_index = 0
st.session_state.prev_search = search
st.write("")

connection = connect_gsheets_api()
project_df = load_data(gsheet_url_proj, project_columns)

if search:     
    search_df = local_search(search, project_df, pub_df)
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

st.write("")
st.write("")

# ANNOTATION FUNCTION

st.header("Annotate")
        
project_id = active_df.iloc[selected_row_index][acc_col]
label_options = set()
if not project_df.empty:
    for l in project_df[annot_col]:
        if l is not None:
            label_options.update(set(l.split(delimiter)))
    label_options = sorted(list(label_options))
    
original_labels = project_df.at[selected_row_index, annot_col]
if original_labels is None:
    original_labels = []
else:
    original_labels = original_labels.split(delimiter)

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

def start_learning():
    return

st.header("Learn")
st.write("Initiate learning algorithm:")
learn_button = st.button("Start", on_click=start_learning)
    
