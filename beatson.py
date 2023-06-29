import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from Bio import Entrez
from collections import defaultdict
from shillelagh.backends.apsw.db import connect

connection = connect(
    ":memory:",
    adapter_kwargs = {
        "gsheetsapi": { 
            "service_account_info":  st.secrets["gcp_service_account"] 
        }
    }
)

Entrez.email = "stell.aeva@hotmail.com"
db = "bioproject"
retmax = 10

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

st.write("Here we go")
search = st.text_input("Search:")
if search:
    search_terms = search.split()
    # find synonyms?
    # entrez spelling suggestions?
    term = "+AND+".join(search_terms)
    handle = Entrez.esearch(db=db, term=term, retmax=retmax, idtype="acc")
    ids = Entrez.read(handle)["IdList"]
    if ids:
        handle = Entrez.efetch(db=db, id=",".join(ids))
        tree = ET.parse(handle)
        tag, data_dict = recursive_dict(tree.getroot())
        # print(tag, data_dict)
        # query
        # display results
    
