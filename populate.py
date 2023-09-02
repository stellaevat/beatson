from features.gsheets import get_gsheets_urls, connect_gsheets_api, batch_store_sheet
from features.search import retrieve_projects

GSHEETS_URL_PROJ, GSHEETS_URL_PUB, GSHEETS_URL_METRICS = get_gsheets_urls()

if __name__ == '__main__':
    connection = connect_gsheets_api()
    ids = ""
    with open("project_ids.txt", encoding="utf8") as f:
        ids = f.readlines()
    all_project_data, all_pub_data = retrieve_projects(ids)
    if all_project_data:
        batch_store_sheet(connection, all_project_data, GSHEETS_URL_PROJ)
    if all_pub_data:
        batch_store_sheet(connection, all_pub_data, GSHEETS_URL_PUB)