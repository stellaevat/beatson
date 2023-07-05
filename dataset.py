import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import time
from Bio import Entrez
from collections import defaultdict
from shillelagh.backends.apsw.db import connect

Entrez.email = "stell.aeva@hotmail.com"
project_db = "bioproject"
pub_db = "pubmed"
retmax = 10
idtype="acc"
rettype="xml"
retmode="xml"
gsheet_url = st.secrets["private_gsheets_url"]
id_col = "UID"
vector_col = "Project_vector"
annot_col = "Project_labels"



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

def efetch(database, ids):
    if not ids:
        return None
    handle = Entrez.efetch(db=database, id=ids, rettype=rettype, retmode=retmode)
    tree = ET.parse(handle)
    tag, data_dict = parse_xml(tree.getroot())   
    return data_dict 
    
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

# For beatson script
def load_annotations(gsheet_url):
    select = f"""
            SELECT * EXCEPT ({vector_col})
            FROM "{gsheet_url}"
            """
    executed_query = connection.execute(select)
    annot_df = pd.DataFrame(executed_query.fetchall())
    if not annot_df.empty:
        annot_df.columns = [id_col, annot_col]
        annot_df.set_index(id_col, inplace=True)
    return annot_df
    
def insert_annotation(gsheet_url, project_uid, sparse, labels):
    insert = f"""
            INSERT INTO "{gsheet_url}" ({id_col}, {vector_col}, {annot_col})
            VALUES ({project_uid}, "{sparse}", "{labels}")
            """
    connection.execute(insert)
    
ids = "431170,988417"
project_dict = efetch(project_db, ids)
projects = project_dict["DocumentSummary"]
if not isinstance(projects, list):
    projects = [projects,]
    
project_data = defaultdict(list)
for project in projects:
    # Validate expected structure
    if "error" not in project and "uid" in project and "Project" in project and "ProjectDescr" in project["Project"]:
        data = project["Project"]["ProjectDescr"]
        project_data["uid"].append(int(project["uid"]))
        text = data.get("Title", "") + " " + data.get("Description", "") + " "
        
        
        pub_refs = data.get("Publication", [])
        pub_ids = ""
        if pub_refs:
            if not isinstance(pub_refs, list):
                pub_refs = [pub_refs,]
            pub_ids = ",".join([pub.get("id") for pub in pub_refs])
            pub_dict = efetch(pub_db, pub_ids)
            publications = pub_dict["PubmedArticle"]
            if not isinstance(publications, list):
                publications = [publications,]
            
            
            for pub in publications:
                if "error" not in pub and "MedlineCitation" in pub and "Article" in pub["MedlineCitation"] and "Abstract" in pub["MedlineCitation"]["Article"] and "AbstractText" in pub["MedlineCitation"]["Article"]["Abstract"]:
                    abstract_pieces = pub["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                    if not isinstance(abstract_pieces, list):
                        abstract_pieces = [abstract_pieces,]
                    for piece in abstract_pieces:
                        if isinstance(piece, str):
                            text += (piece + " ")
                        elif isinstance(piece, dict) and "text" in piece:
                            text += (piece["text"] + " ")

        project_data["text"].append(text)
        time.sleep(2)
        
print(len(project_data["text"][0]))        
print(project_data["text"][0])
print(len(project_data["text"][1]))  
print(project_data["text"][1])
                
                            
                            
        