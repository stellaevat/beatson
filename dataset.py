import streamlit as st
import xml.etree.ElementTree as ET
import time
import re
from Bio import Entrez
from collections import defaultdict
from shillelagh.backends.apsw.db import connect

GSHEET_URL_PROJ = st.secrets["private_gsheets_url_proj"]
GSHEET_URL_PUB = st.secrets["private_gsheets_url_pub"]

Entrez.email = "stell.aeva@hotmail.com"
PROJECT_DB = "bioproject"
PUB_DB = "pubmed"

DELIMITER = ", "
RETTYPE = "xml"
RETMODE = "xml"
ENTREZ_API_CALLS_PS = 3
GOOGLE_API_CALLS_PM = 60

project_columns = ["UID", "Accession", "Title", "Name", "Description", "Data_Type", "Scope", "Organism", "PMIDs", "Annotation", "Predicted", "Probability", "To_Annotate"]

style_tags = ["b", "i", "p"]

@st.cache_resource(show_spinner=False)
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

def store_data(gsheet_url, entries):
    columns = list(entries[0].keys())
    values = []
    for entry in entries:
        entry_str = '("' + '", "'.join([entry[col] for col in columns]) + '")'
        values.append(entry_str)
        
    for batch in range(0, len(values), GOOGLE_API_CALLS_PM):
        values_str = ",\n".join(values[batch : min(batch + GOOGLE_API_CALLS_PM, len(values))])
        insert = f''' 
                INSERT INTO "{gsheet_url}" ({", ".join(columns)})
                VALUES {values_str}
                '''
        connection.execute(insert)
        print(f"{min(batch + GOOGLE_API_CALLS_PM, len(values))}/{len(values)} entries stored...")
        # Not to exceed API limit
        time.sleep(60)
        
 
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
def efetch(database, ids):
    if not ids:
        return None
    handle = Entrez.efetch(db=database, id=ids, rettype=RETTYPE, retmode=RETMODE)
    tree = ET.parse(handle)
    tag, data_dict = parse_xml(tree.getroot())
    return data_dict 
    
@st.cache_data(show_spinner=False) 
def clean_text(data_dict):
    for col, data in data_dict.items():
        if "<" in data and ">" in data:
            for tag in style_tags:
                data = data.replace(f"<{tag}>", " ")
                data = data.replace(f"<{tag.upper()}>", " ")
                data = data.replace(f"</{tag}>", " ")
                data = data.replace(f"</{tag.upper()}>", " ")
        
        # TODO: If quotes useful, replace with single quote instead
        # For API call syntax purposes
        data = data.replace('"', "").strip()
        if len(data) > 1:
            data = data[0].upper() + data[1:]
        else:
            data = data.upper()
        
        data_dict[col] = data
    return data_dict
    
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

# connection = connect_gsheets_api()
# ids = ""
# with open("random_ids.txt", encoding="utf8") as f:
    # ids = f.readlines()
# all_project_data, all_pub_data = retrieve_projects(ids)
# if all_project_data:
    # store_data(GSHEET_URL_PROJ, all_project_data)
# if all_pub_data:
    # store_data(GSHEET_URL_PUB, all_pub_data)