import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from Bio import Entrez
from collections import defaultdict
from shillelagh.backends.apsw.db import connect

st.set_page_config(page_title="BioProject Annotation")

Entrez.email = "stell.aeva@hotmail.com"
database = "bioproject"
base_url = "https://www.ncbi.nlm.nih.gov/bioproject/"
max_search_results = 10
id_col = "UID"
annot_col = "Project_labels"
project_col = "Project_title"
url_col = "URL"
delimiter = ", "
css = r'''
    <style>
        [data-testid="stForm"] {border: 0px; padding: 0px;}
        [kind="secondaryFormSubmit"] {position: absolute; right: 0px;}
    </style>
'''
st.markdown(css, unsafe_allow_html=True)

@st.cache_data(show_spinner="Getting search results...")
def esearch(db=database, term="", retmax=max_search_results, idtype="acc"):
    if not term:
        return None
    handle = Entrez.esearch(db=db, term=term, retmax=retmax, idtype=idtype)
    ids = Entrez.read(handle)["IdList"]
    return ids
    
@st.cache_data(show_spinner="Getting search results...")
def efetch(db=database, id="", rettype="xml", retmode="xml"):
    if not id:
        return None, None
    handle = Entrez.efetch(db=db, id=",".join(ids), rettype=rettype, retmode=retmode)
    tree = ET.parse(handle)
    tag, project_dict = parse_xml(tree.getroot())
    return project_dict


@st.cache_resource(show_spinner="Loading project data...")
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

@st.cache_resource(show_spinner="Loading project data...")
def load_annotations(gsheet_url):
    query = f'SELECT * FROM "{gsheet_url}"'
    executed_query = connection.execute(query)
    annot_df = pd.DataFrame(executed_query.fetchall())
    annot_df.columns = [id_col, annot_col]
    annot_df[url_col] = list(map(uid_to_url, annot_df[id_col]))
    annot_df.set_index(id_col, inplace=True)
    return annot_df
    
    
def insert_annotation(gsheet_url, project_uid, labels):
    insert = f"""
            INSERT INTO "{gsheet_url}" ({id_col}, {annot_col})
            VALUES ({project_uid}, "{labels}")
            """
    connection.execute(insert)
    
def update_annotation(gsheet_url, project_uid, labels):
    update = f"""
            UPDATE "{gsheet_url}"
            SET {annot_col} = "{labels}"
            WHERE {id_col} = {project_uid}
            """
    connection.execute(update)
    
def submit_labels():
    # TODO: determine what happens if new label set is empty
    new = st.session_state.get("new", "")
    labels = st.session_state.get("labels", "")
    project_uid = st.session_state.get("project_uid", "")
    
    if project_uid and (labels or new):
        project_uid = int(project_uid)
        combined = (set(labels) | {l.strip() for l in new.split(",")}) - {""}
        labels_str = delimiter.join(sorted(list(combined)))
        if project_uid not in annot_df.index:
            insert_annotation(gsheet_url, project_uid, labels_str)
            annot_df.loc[project_uid] = [labels_str, uid_to_url(project_uid)]
        elif combined ^ set(original_labels):
            update_annotation(gsheet_url, project_uid, labels_str)
            annot_df.at[project_uid, annot_col] = labels_str
            
    st.session_state.new = ""

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
    
def uid_to_url(uid):
    return base_url + str(uid) + "/"
    
st.title("BioProject Annotation")

st.header("Search")
search = st.text_input("Search:", label_visibility="collapsed")
search_df = None

if search:
    search_terms = search.split()
    # TODO: find synonyms / entrez spelling suggestions?
    ids = esearch(term="+AND+".join(search_terms))
    if ids:
        project_dict = efetch(id=",".join(ids))
        projects = project_dict['DocumentSummary']
        if not isinstance(projects, list):
            projects = [projects,]
        uids = [int(project["uid"]) for project in projects]
        titles = [project["Project"]["ProjectDescr"]["Title"] for project in projects]
        urls = list(map(uid_to_url, uids))
        search_df = pd.DataFrame({id_col:uids, project_col:titles, url_col:urls}, columns=(id_col, project_col, url_col))
        search_df.set_index(id_col, inplace=True)
        st.dataframe(
            search_df,
            use_container_width=True,
            column_config={
                id_col: st.column_config.TextColumn(),
                project_col: st.column_config.TextColumn(),
                url_col: st.column_config.LinkColumn(width="small", validate="^"+base_url+"[0-9]*/$")
            },
        )

st.header("Annotate")
connection = connect_gsheets_api()
gsheet_url = st.secrets["private_gsheets_url"]
annot_df = load_annotations(gsheet_url)
col1, col2 = st.columns(2)
    
with col1:
    project_options = set(annot_df.index.tolist())
    if search_df is not None:
        project_options.update(set(search_df.index.tolist()))
    project_uid = st.selectbox("Project UID:", [""] + sorted(list(project_options)), key="project_uid")
    if project_uid:
        project_uid = int(project_uid)

with col2:
    label_options = set()
    for l in annot_df[annot_col]:
        label_options.update(set(l.split(delimiter)))
    label_options = sorted(list(label_options))
        
    original_labels = None
    if project_uid in annot_df.index:
        original_labels = annot_df.at[project_uid, annot_col].split(delimiter)
        
    with st.form(key="Annotate"):
        labels = st.multiselect("Labels:", label_options, default=original_labels, key="labels")
        new = st.text_input("or add new:", placeholder="Or type new (comma-separated)", label_visibility="collapsed", key="new")
        submit_button = st.form_submit_button("Submit", on_click=submit_labels)
        
        
st.header("Review")
st.dataframe(
    annot_df,
    use_container_width=True,
    column_config={
        id_col: st.column_config.TextColumn(),
        annot_col: st.column_config.TextColumn(),
        url_col: st.column_config.LinkColumn(width="small", validate="^"+base_url+"[0-9]*/$")
    },
)
