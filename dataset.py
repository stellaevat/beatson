import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import time
from Bio import Entrez
from collections import defaultdict
from shillelagh.backends.apsw.db import connect

Entrez.email = "stell.aeva@hotmail.com"
api_calls_ps = 3
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
delimiter = ", "

@st.cache_resource
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


@st.cache_data
def efetch(database, ids):
    if not ids:
        return None
    handle = Entrez.efetch(db=database, id=ids, rettype=rettype, retmode=retmode)
    tree = ET.parse(handle)
    tag, data_dict = parse_xml(tree.getroot())
    return data_dict 
    
@st.cache_data    
def get_project_data(project):
    project_data = {}
    
    archive = project.get("ProjectID",[{}])[0].get("ArchiveID", [{}])[0].get("attrs", {})
    project_descr = project.get("ProjectDescr", [{}])[0]
    project_type = project.get("ProjectType", [{}])[0].get("ProjectTypeSubmission", [{}])[0]
    target = project_type.get("Target", [{}])[0]
    publications = project_descr.get("Publication", [])
    
    project_data["uid"] = archive.get("id", "")
    project_data["acc"] = archive.get("accession", "")
    
    project_data["title"] = project_descr.get("Title", "")
    project_data["name"] = project_descr.get("Name", "")
    project_data["description"] = project_descr.get("Description", "")
    
    project_data["datatype"] = project_type.get("ProjectDataTypeSet", [{}])[0].get("DataType", "")
    project_data["scope"] = target.get("attrs", {}).get("sample_scope", "")
    project_data["organism"] = target.get("Organism", [{}])[0].get("OrganismName", "") 
    
    project_data["publications"] = ",".join([pub.get("attrs", {})["id"] for pub in publications if "id" in pub.get("attrs", {})])
    return project_data

@st.cache_data    
def get_publication_data(pub):
    pub_data = {}
    article = pub.get("Article", [{}])[0]
    
    pub_data["pmid"] = pub.get("PMID", [{}])[0].get("text", "")
    pub_data["title"] = article.get("ArticleTitle", "")
    
    clean = []
    pieces = article.get("Abstract", [{}])[0].get("AbstractText", [])
    if isinstance(pieces, str):
        clean.append(pieces)
    else:
        for piece in pieces:
            if isinstance(piece, str):
                clean.append(piece)
            elif isinstance(piece, dict) and "text" in piece:
                clean.append(piece["text"])
    pub_data["abstract"] = " ".join(clean)
    
    mesh_section = pub.get("MeshHeadingList", [{}])[0].get("MeshHeading", [])
    mesh_list = []
    for descriptor in mesh_section:
        name = descriptor.get("DescriptorName", [{}])[0].get("text", "")
        qualifier = descriptor.get("QualifierName", [{}])[0].get("text", "")
        if name:
            mesh_list.append(name)
        if qualifier:
            mesh_list.append(qualifier)
    pub_data["mesh"] = delimiter.join(mesh_list)
            
    key_section = pub.get("KeywordList", [{}])[0].get("Keyword", [])
    keyword_list = [keyword["text"] for keyword in key_section if "text" in keyword]
    pub_data["keywords"] = delimiter.join(keyword_list)
    
    return pub_data
    
@st.cache_data
def retrieve_projects(ids):
    if ids:
        project_dict = efetch(project_db, ids)
        api_calls = 1
        
        if (projects := project_dict.get("DocumentSummary")):
            all_project_data, all_pub_data = [], []
            
            for project in projects:
                if "Project" in project:
                    project_data = get_project_data(project["Project"][0])
                    all_project_data.append(project_data)
                    
                    pub_ids = project_data["publications"]
                    if pub_ids:
                        pub_dict = efetch(pub_db, pub_ids)
                        api_calls += 1
                        
                        if (publications := pub_dict.get("PubmedArticle")):
                            for pub in publications:
                                if "MedlineCitation" in pub:
                                    pub_data = get_publication_data(pub["MedlineCitation"][0])
                                    all_pub_data.append(pub_data)
                                    
                    # Not to exceed API limit
                    if api_calls % api_calls_ps == 0:
                        time.sleep(1)
                        
            return all_project_data, all_pub_data
        else:
            print("Error retrieving projects.")
    else:
        print("No project ids given.")
    