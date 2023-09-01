import streamlit as st
from gsheets import get_gsheets_urls, connect_gsheets_api, batch_store_sheet
from search import retrieve_projects

GSHEETS_URL_PROJ, GSHEETS_URL_PUB, GSHEETS_URL_METRICS = get_gsheets_urls()

if __name__ == '__main__':
    connections = [connect_gsheets_api(i) for i in range(5)]
    st.session_state.connection_used = 0
    ids = ""
    with open("project_ids.txt", encoding="utf8") as f:
        ids = f.readlines()
    all_project_data, all_pub_data = retrieve_projects(ids)
    if all_project_data:
        batch_store_sheet(connections, all_project_data, GSHEETS_URL_PROJ)
    if all_pub_data:
        batch_store_sheet(connections, all_pub_data, GSHEETS_URL_PUB)