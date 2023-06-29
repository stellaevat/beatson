import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from Bio import Entrez
from collections import defaultdict
from shillelagh.backends.apsw.db import connect

Entrez.email = "stell.aeva@hotmail.com"
database = "bioproject"
id_col = "project_id"
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

@st.cache_data
def get_annotations(sheet_url):
    query = f'SELECT * FROM "{sheet_url}"'
    executed_query = connection.execute(query)
    df = pd.DataFrame(executed_query.fetchall())
    df.columns = [id_col, annot_col]
    df.set_index(id_col, inplace=True)
    return df
    
def add_annotation(sheet_url, project_id, labels):
    insert = f"""
            INSERT INTO "{sheet_url}" ({id_col}, {annot_col})
            VALUES ("{project_id}", "{labels}")
            """
    connection.execute(insert)
    
def update_annotation(sheet_url, project_id, labels):
    update = f"""
            UPDATE "{sheet_url}"
            SET {annot_col} = "{labels}"
            WHERE {id_col} = "{project_id}"
            """
    connection.execute(update)
    
def clean_labels():
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


st.write("Here we go")
search = st.text_input("Search:")
if search:
    search_terms = search.split()
    # find synonyms?
    # entrez spelling suggestions?
    ids = esearch(term="+AND+".join(search_terms))
    if ids:
        tag, data_dict = efetch(id=",".join(ids))
        if tag:
            print(tag, data_dict)
        # display results

connection = connect_gsheet_api()
sheet_url = st.secrets["private_gsheets_url"]
df = get_annotations(sheet_url)

st.header("Annotate")
col1, col2 = st.columns(2)

with col1:
    project_id = st.text_input("Project ID:")
# if project_id:
    # project_id = int(project_id)

if project_id in df.index:
    labels_old = df.at[project_id, annot_col]
    with col2:
        labels = st.text_input("Labels:", value=labels_old, key="labels", on_change=clean_labels)
    if labels != labels_old:
        update_annotation(sheet_url, project_id, labels)
        df.at[project_id, annot_col] = labels
else:
    with col2:
        labels = st.text_input("Labels:", key="labels", on_change=clean_labels)
    if project_id and labels:
        add_annotation(sheet_url, project_id, labels)
        df.loc[project_id] = [labels,]

st.button("Submit")

st.header("Samples")
st.write(df)
