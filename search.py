import streamlit as st
from Bio import Entrez
import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from gsheets import connect_gsheets_api, batch_store_sheet, get_gsheets_urls, get_gsheets_columns, get_delimiter

Entrez.email = "stell.aeva@hotmail.com"
ENTREZ_API_CALLS_PS = 3

PROJECT_DB = "bioproject"
PUB_DB = "pubmed"
BASE_URL_PROJ = "https://www.ncbi.nlm.nih.gov/" + PROJECT_DB + "/"
BASE_URL_PUB = "https://pubmed.ncbi.nlm.nih.gov/"

IDTYPE = "acc"
RETTYPE = "xml"
RETMODE = "xml"
RETMAX = 10

project_columns, pub_columns, metric_columns = get_gsheets_columns()
style_tags = ["b", "i", "p"]
DELIMITER = get_delimiter()

search_msg = "Getting search results..."

def get_databases():
    return PROJECT_DB, PUB_DB
    
def get_base_urls():
    return BASE_URL_PROJ, BASE_URL_PUB

def parse_xml(element):
    is_text = (element.text and not element.text.isspace())
    if not is_text:
        for child in element:
            if child.tail and not child.tail.isspace():
                is_text = True
                break
    
    data_dict = {}    
    if is_text:
        text = element.text if (element.text and not element.text.isspace()) else ""
        for child in element:
            # TODO: Remove spaces if sub/super-scripts used in biomedicine terminology
            text += (" " + child.text  + " ") if (child.text and not child.text.isspace()) else ""
            text += child.tail if (child.tail and not child.tail.isspace()) else ""
        data_dict["text"] = text
    else:
        children = map(parse_xml, element)
        children_nodes = defaultdict(list)
        for node, data in children:
            children_nodes[node].append(data)
        for node, data_list in children_nodes.items():
            if len(data_list) == 1 and isinstance(data_list[0], str):
                children_nodes[node] = data_list[0]
        if children_nodes:
            data_dict.update(children_nodes)
    
    if element.attrib:
        data_dict["attrs"] = dict(element.attrib)
        
    if len(data_dict) == 1 and "text" in data_dict:
        data_dict = data_dict["text"]
        
    tag = element.tag
    return tag, data_dict

@st.cache_data(show_spinner=False) 
def clean_text(data_dict):
    for col, data in data_dict.items():
        if "<" in data and ">" in data:
            for tag in style_tags:
                data = data.replace(f"<{tag}>", " ")
                data = data.replace(f"<{tag.upper()}>", " ")
                data = data.replace(f"</{tag}>", " ")
                data = data.replace(f"</{tag.upper()}>", " ")

        # For API call syntax purposes
        data = data.replace('"', "").strip()
        if len(data) > 1:
            data = data[0].upper() + data[1:]
        else:
            data = data.upper()
        
        data_dict[col] = data
    return data_dict
    
@st.cache_data(show_spinner=False) 
def efetch(database, ids):
    if not ids:
        return None
    handle = Entrez.efetch(db=database, id=ids, rettype=RETTYPE, retmode=RETMODE)
    tree = ET.parse(handle)
    tag, data_dict = parse_xml(tree.getroot())
    return data_dict 

@st.cache_data(show_spinner=False)
def esearch(database, terms):
    if not terms:
        return None
    handle = Entrez.esearch(db=database, term=terms, retmax=RETMAX, idtype=IDTYPE)
    data_dict = Entrez.read(handle)
    ids = data_dict["IdList"]
    return ids
    
@st.cache_data(show_spinner=False)   
def get_project_data(project):
    project_data = {}
    
    archive = project.get("ProjectID",[{}])[0].get("ArchiveID", [{}])[0].get("attrs", {})
    project_descr = project.get("ProjectDescr", [{}])[0]
    project_type = project.get("ProjectType", [{}])[0].get("ProjectTypeSubmission", [{}])[0]
    target = project_type.get("Target", [{}])[0]
    publications = project_descr.get("Publication", [])
    
    project_data["UID"] = archive.get("id", "")
    project_data["Accession"] = archive.get("accession", "")
    
    project_data["Title"] = project_descr.get("Title", "")
    project_data["Name"] = project_descr.get("Name", "")
    project_data["Description"] = project_descr.get("Description", "")
    
    project_data["Data_Type"] = project_type.get("ProjectDataTypeSet", [{}])[0].get("DataType", "")
    project_data["Scope"] = target.get("attrs", {}).get("sample_scope", "")
    if project_data["Scope"]:
        project_data["Scope"] = project_data["Scope"][1:]
    project_data["Organism"] = target.get("Organism", [{}])[0].get("OrganismName", "") 
    
    pub_list = []
    for pub in publications:
        if "id" in pub.get("attrs", {}):
            pub_id = pub["attrs"]["id"].strip()
            if pub_id.isnumeric():
                pub_list.append(pub_id)
    project_data["PMIDs"] = DELIMITER.join(pub_list)
    
    for column in project_columns:
        if column not in project_data: 
            project_data[column] = ""
    
    project_data = clean_text(project_data)
    return project_data

@st.cache_data(show_spinner=False)    
def get_publication_data(pub):
    pub_data = {}
    article = pub.get("Article", [{}])[0]
    
    pub_data["PMID"] = pub.get("PMID", [{}])[0].get("text", "")
    pub_data["Title"] = article.get("ArticleTitle", "")
    
    clean = []
    pieces = article.get("Abstract", [{}])[0].get("AbstractText", [])
    if isinstance(pieces, str):
        clean.append(pieces)
    else:
        for piece in pieces:
            if isinstance(piece, str):
                clean.append(piece.strip())
            elif isinstance(piece, dict) and "text" in piece:
                clean.append(piece["text"].strip())
    pub_data["Abstract"] = " ".join(clean)
    
    mesh_section = pub.get("MeshHeadingList", [{}])[0].get("MeshHeading", [])
    mesh_list = []
    for descriptor in mesh_section:
        name = descriptor.get("DescriptorName", [{}])[0].get("text", "")
        qualifier = descriptor.get("QualifierName", [{}])[0].get("text", "")
        if name:
            mesh_list.append(name.strip())
        if qualifier:
            mesh_list.append(qualifier.strip())
    pub_data["MeSH"] = DELIMITER.join(mesh_list)
            
    key_section = pub.get("KeywordList", [{}])[0].get("Keyword", [])
    keyword_list = [keyword["text"].strip() for keyword in key_section if "text" in keyword]
    pub_data["Keywords"] = DELIMITER.join(keyword_list)
    
    pub_data = clean_text(pub_data)
    return pub_data
    
@st.cache_data(show_spinner=False)
def retrieve_projects(ids):
    all_project_data, all_pub_data = [], []
    all_pub_ids = set()
    
    if ids:
        project_dict = efetch(PROJECT_DB, ids)
        
        if (projects := project_dict.get("DocumentSummary")):
            for project in projects:
                if "Project" in project:
                    project_data = get_project_data(project["Project"][0])
                    all_project_data.append(project_data)
                    
                    pub_ids = project_data["PMIDs"]
                    if pub_ids:
                        all_pub_ids.update(pub_ids.split(DELIMITER))
            
            if all_pub_ids:
                pub_dict = efetch(PUB_DB, all_pub_ids)
                
                if (publications := pub_dict.get("PubmedArticle")):
                    for pub in publications:
                        if "MedlineCitation" in pub:
                            pub_data = get_publication_data(pub["MedlineCitation"][0])
                            all_pub_data.append(pub_data)
        else:
            print("Error retrieving projects.")
    else:
        print("No project ids given.")
    
    return all_project_data, all_pub_data
    
    
 
@st.cache_resource(show_spinner=search_msg)
def api_search(search_terms, existing_projects):
    ids = esearch(PROJECT_DB, search_terms)
    ids_to_fetch = [project_id for project_id in ids if project_id not in existing_projects] # either uid or acc because esearch unreliable

    search_df, search_pub_df = None, None
    all_project_data, all_pub_data = retrieve_projects(ids_to_fetch)
    if all_project_data:
        search_df = pd.DataFrame(all_project_data)
        search_df.columns = project_columns
        if all_pub_data:
            search_pub_df = pd.DataFrame(all_pub_data)
            search_pub_df.columns = pub_columns
    
    found = True if ids else False    
    return search_df, search_pub_df, found
            
            
@st.cache_resource(show_spinner=search_msg)
def local_search(search_terms, df, text_columns):
    search_terms = {term.strip().lower() for term in search_terms.split() if term.strip()}
    search_expr = "".join([f"(?=.*{term})" for term in search_terms])
    
    raw_counts = np.column_stack([df.astype(str)[col].str.count(search_expr, flags=re.IGNORECASE) for col in text_columns])
    total_counts = np.sum(raw_counts, axis=1)
    mask = np.where(total_counts > 0, True, False)
    search_df = df.loc[mask]
    
    search_df = search_df.sort_index(axis=0, key=lambda col: col.map(lambda i: total_counts[i]), ascending=False, ignore_index=True)
    return search_df
    
def display_search_feature(tab):
    search_terms = st.text_input("Search", label_visibility="collapsed", placeholder="Search", autocomplete="", key=(tab + "_search")).strip()
    st.write("")
    
    if st.session_state.get(tab + "_prev_search", "") != search_terms:
        st.session_state[tab + "_selected_row_index"] = 0
        st.session_state[tab + "_selected_projects"] = []
    st.session_state[tab + "_prev_search"] = search_terms
    
    return search_terms