import re
import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from Bio import Entrez
from collections import defaultdict
from shillelagh.backends.apsw.db import connect

st.set_page_config(page_title="BioProject Annotation")

Entrez.email = "stell.aeva@hotmail.com"
database = "bioproject"
base_url = "https://www.ncbi.nlm.nih.gov/bioproject/"
max_search_results = 10
search_msg = "Getting search results..."
loading_msg = "Loading project data..."
gsheet_url_proj = st.secrets["private_gsheets_url_proj"]
gsheet_url_pub = st.secrets["private_gsheets_url_pub"]
gsheet_url_annot = st.secrets["private_gsheets_url_annot"]
project_columns = ["uid", "acc", "title", "name", "description", "datatype", "scope", "organism", "publications", "labels"]
pub_columns = ["pmid", "title", "abstract", "mesh", "keywords"]
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

def uid_to_url(uid):
    return base_url + str(uid) + "/"
    
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
def load_annotations(gsheet_url_annot):
    query = f'SELECT * FROM "{gsheet_url_annot}"'
    executed_query = connection.execute(query)
    annot_df = pd.DataFrame(executed_query.fetchall())
    if not annot_df.empty:
        annot_df.columns = [id_col, annot_col]
        annot_df[url_col] = list(map(uid_to_url, annot_df[id_col]))
        annot_df.set_index(id_col, inplace=True)
    return annot_df
    
def insert_annotation(gsheet_url_annot, project_uid, labels):
    insert = f"""
            INSERT INTO "{gsheet_url_annot}" ({id_col}, {annot_col})
            VALUES ({project_uid}, "{labels}")
            """
    connection.execute(insert)
    
def update_annotation(gsheet_url_annot, project_uid, labels):
    update = f"""
            UPDATE "{gsheet_url_annot}"
            SET {annot_col} = "{labels}"
            WHERE {id_col} = {project_uid}
            """
    connection.execute(update)
    
def delete_annotation(gsheet_url_annot, project_uid):
    delete = f"""
            DELETE FROM "{gsheet_url_annot}"
            WHERE {id_col} = {project_uid}
            """
    connection.execute(delete) 
    
@st.cache_resource(show_spinner=loading_msg)
def load_data(gsheet_url, _columns):
    query = f'SELECT * FROM "{gsheet_url}"'
    executed_query = connection.execute(query)
    df = pd.DataFrame(executed_query.fetchall())
    if not df.empty:
        df.columns = _columns
    return df
    
def submit_labels():
    new = st.session_state.get("new", "")
    labels = st.session_state.get("labels", "")
    project_uid = st.session_state.get("project_uid", "")
    
    if project_uid:
        project_uid = int(project_uid)
        
        if labels or new:
            combined = (set(labels) | {l.strip() for l in new.split(",")}) - {""}
            labels_str = delimiter.join(sorted(list(combined)))
            if annot_df.empty:
                insert_annotation(gsheet_url_annot, project_uid, labels_str)
                annot_df[id_col] = [project_uid,]
                annot_df[annot_col] = [labels_str,]
                annot_df[url_col] = [uid_to_url(project_uid),]
                annot_df.set_index(id_col, inplace=True)
            elif project_uid not in annot_df.index:
                insert_annotation(gsheet_url_annot, project_uid, labels_str)
                annot_df.loc[project_uid] = [labels_str, uid_to_url(project_uid)]
            elif combined ^ set(original_labels):
                update_annotation(gsheet_url_annot, project_uid, labels_str)
                annot_df.at[project_uid, annot_col] = labels_str
                
        elif project_uid in annot_df.index:
            delete_annotation(gsheet_url_annot, project_uid)
            annot_df.drop(project_uid, inplace=True)
            
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
                uids.append(int(project["uid"]))
                titles.append(project["Project"]["ProjectDescr"]["Title"])
                urls.append(uid_to_url(project["uid"]))
        if uids:
            search_df = pd.DataFrame(
                {id_col:uids, project_col:titles, url_col:urls}, 
                columns=(id_col, project_col, url_col)
            )
            search_df.set_index(id_col, inplace=True)
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
    
    search_df.sort_index(axis=0, key=lambda col: col.map(lambda i: total_counts[i]), ascending=False, inplace=True, ignore_index=True)
    return search_df

    
st.title("BioProject Annotation")

st.header("Search")
search = st.text_input("Search:", label_visibility="collapsed")
connection = connect_gsheets_api()
project_df = load_data(gsheet_url_proj, project_columns)
pub_df = load_data(gsheet_url_pub, pub_columns)

if search:
    search_df = local_search(search, project_df, pub_df)
    if search_df is not None and not search_df.empty:
        st.write(f"Search results for '{search}':")
        st.dataframe(
            search_df,
            use_container_width=True,
            hide_index=True,
            column_config={col : st.column_config.TextColumn() for col in project_columns},
        )
    else:
        st.write("No search results.")
        st.dataframe(
            project_df,
            use_container_width=True,
            hide_index=True,
            column_config={col : st.column_config.TextColumn() for col in project_columns},
        )
else:
    st.dataframe(
        project_df,
        use_container_width=True,
        hide_index=True,
        column_config={col : st.column_config.TextColumn() for col in project_columns},
    )

st.header("Annotate")


# annot_df = load_annotations(gsheet_url_annot)
# col1, col2 = st.columns(2)
    
# with col1:
    # project_options = set(annot_df.index.tolist())
    # if search_df is not None:
        # project_options.update(set(search_df["id_col"].tolist()))
    # project_uid = st.selectbox("Project UID:", [""] + sorted(list(project_options)), key="project_uid")
    # if project_uid:
        # project_uid = int(project_uid)

# with col2:
    # label_options = set()
    # if not annot_df.empty:
        # for l in annot_df[annot_col]:
            # label_options.update(set(l.split(delimiter)))
        # label_options = sorted(list(label_options))
        
    # original_labels = None
    # if project_uid in annot_df.index:
        # original_labels = annot_df.at[project_uid, annot_col].split(delimiter)
        
    # with st.form(key="Annotate"):
        # if label_options:
            # labels = st.multiselect("Labels:", label_options, default=original_labels, key="labels")
            # new = st.text_input("Or create new:", placeholder="Or create new (comma-separated)", label_visibility="collapsed", key="new")
        # else:
            # labels = ""
            # new = st.text_input("Labels:", placeholder="Create new (comma-separated)", key="new")
        # submit_button = st.form_submit_button("Submit", on_click=submit_labels)
        

# def start_learning():
    # return

# if not annot_df.empty:      
    # st.header("Review")
    # st.write(f"{annot_df.shape[0]} project{'' if annot_df.shape[0] == 1 else 's'} annotated. {len(label_options)} disctinct label{'' if len(label_options) == 1 else 's'} used.")
    # st.dataframe(
        # annot_df,
        # use_container_width=True,
        # column_config={
            # id_col: st.column_config.TextColumn(),
            # annot_col: st.column_config.TextColumn(),
            # url_col: st.column_config.LinkColumn(width="small", validate="^"+base_url+"[0-9]*/$")
        # },
    # )
    
    # st.header("Learn")
    # st.write("This does nothing.")
    # learn_button = st.button("Start", on_click=start_learning)
    
