import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from Bio import Entrez
from collections import defaultdict
from shillelagh.backends.apsw.db import connect

Entrez.email = "stell.aeva@hotmail.com"
database = "bioproject"
id_col = "uid"
annot_col = "labels"

@st.cache_resource
def connect_gsheet_api():
    connection = connect(
        ":memory:",
        adapter_kwargs = {
            "gsheetsapi": { 
                "service_account_info":  dict(st.secrets["gcp_service_account"])
            }
        }
    )
    return connection

@st.cache_resource
def get_annotations(gsheet_url):
    query = f'SELECT * FROM "{gsheet_url}"'
    executed_query = connection.execute(query)
    annot_df = pd.DataFrame(executed_query.fetchall())
    annot_df.columns = [id_col, annot_col]
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
    
def clean_labels():
    # Will need to supress warning once option available
    if st.session_state.get("labels"):
        val = st.session_state.labels
        val_clean = [label.strip() for label in val.split(",")]
        st.session_state.labels = ", ".join(val_clean)

def recursive_dict(element):
    data_dict = dict(element.attrib)
    children = map(recursive_dict, element)
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

@st.cache_data
def esearch(db=database, term="", retmax=10, idtype="acc"):
    if not term:
        return None
    handle = Entrez.esearch(db=db, term=term, retmax=retmax, idtype=idtype)
    ids = Entrez.read(handle)["IdList"]
    return ids
    
@st.cache_data
def efetch(db=database, id="", rettype="xml", retmode="xml"):
    if not id:
        return None, None
    handle = Entrez.efetch(db=db, id=",".join(ids), rettype=rettype, retmode=retmode)
    tree = ET.parse(handle)
    tag, data_dict = recursive_dict(tree.getroot())
    return tag, data_dict


st.header("Search")
search = st.text_input("Search:", label_visibility="collapsed")
search_df = None

if search:
    search_terms = search.split()
    # find synonyms / entrez spelling suggestions?
    ids = esearch(term="+AND+".join(search_terms))
    if ids:
        tag, data_dict = efetch(id=",".join(ids))
        projects = data_dict['DocumentSummary']
        if not isinstance(projects, list):
            projects = [projects,]
        uids = [project["uid"] for project in projects]
        titles = [project["Project"]["ProjectDescr"]["Title"] for project in projects]
        search_df = pd.DataFrame({"uid":uids, "title":titles}, columns=("uid", "title"))
        search_df.set_index("uid", inplace=True)
        st.write(search_df)

connection = connect_gsheet_api()
gsheet_url = st.secrets["private_gsheets_url"]
annot_df = get_annotations(gsheet_url)

st.header("Annotate")
col1, col2 = st.columns(2)
    
with col1:
    options = annot_df.index.tolist() if search_df is None else annot_df.index.tolist() + search_df.index.tolist()
    project_uid = st.selectbox("Project UID:", options)
    if project_uid:
        project_uid = int(project_uid)
    
with col2:
    if project_uid in annot_df.index:
        labels = st.text_input("Labels:", value=annot_df.at[project_uid, annot_col], key="labels", on_change=clean_labels)
        if labels and labels != annot_df.at[project_uid, annot_col]:
            update_annotation(gsheet_url, project_uid, labels)
            annot_df.at[project_uid, annot_col] = labels
    else:
        labels = st.text_input("Labels:", value="", key="labels", on_change=clean_labels)
        if labels and project_uid:
            insert_annotation(gsheet_url, project_uid, labels)
            annot_df.loc[project_uid] = [labels,]

st.button("Submit")

st.header("Samples")
st.write(annot_df)
