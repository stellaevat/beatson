import streamlit as st
import pandas as pd
from shillelagh.backends.apsw.db import connect

GSHEETS_URL_PROJ = st.secrets["private_gsheets_url_proj"]
GSHEETS_URL_PUB = st.secrets["private_gsheets_url_pub"]
GSHEETS_URL_METRICS = st.secrets["private_gsheets_url_metrics"]
GSHEET_API_CALLS_PM = 60

ACC_COL = "Accession"
project_columns = ["UID", ACC_COL, "Title", "Name", "Description", "Data_Type", "Scope", "Organism", "PMIDs", "Annotation", "Predicted", "Score", "To_Annotate"]
pub_columns = ["PMID", "Title", "Abstract", "MeSH", "Keywords"]
metric_columns = ["Date", "Dataset_Hash", "Train_Size", "Test_Size", "F1_micro", "F1_macro"]

loading_msg = "Loading project data..."
       
def get_gsheets_urls():
    return GSHEETS_URL_PROJ, GSHEETS_URL_PUB, GSHEETS_URL_METRICS
    
def get_gsheets_columns():
    return project_columns, pub_columns, metric_columns
    
@st.cache_data(show_spinner=False)        
def get_gsheets_urls():
    return GSHEETS_URL_PROJ, GSHEETS_URL_PUB, GSHEETS_URL_METRICS

@st.cache_resource(show_spinner=loading_msg)
def connect_gsheets_api(service_no):
    connection = connect(
        ":memory:",
        adapter_kwargs = {
            "gsheetsapi": { 
                "service_account_info":  dict(st.secrets[f"gcp_service_account_{service_no}"])
            }
        }
    )
    return connection
    
@st.cache_resource(show_spinner=loading_msg)
def load_sheet(_connection, columns, sheet=GSHEETS_URL_PROJ):
    query = f'SELECT * FROM "{sheet}"'
    executed_query = _connection.execute(query)
    df = pd.DataFrame(executed_query.fetchall())
    if not df.empty:
        df.columns = columns
    else:
        df = pd.DataFrame(columns=columns)
    return df
    
def update_sheet(connection, project_id, column_values_dict, sheet=GSHEETS_URL_PROJ):
    column_values = [f'{col} = "{val}"' if val else f'{col} = NULL' for (col, val) in column_values_dict.items()]
    update = f"""
        UPDATE "{sheet}"
        SET {", ".join(column_values)}
        WHERE {ACC_COL} = "{project_id}"
    """
    connection.execute(update)

def insert_sheet(connection, values, columns=project_columns, sheet=GSHEETS_URL_PROJ):
    values_str = '("' + '", "'.join([str(val) for val in values]) + '")'
    insert = f"""
        INSERT INTO "{sheet}" ({", ".join(columns)})
        VALUES {values_str}
    """
    connection.execute(insert)
   
def clear_sheet_column(connection, column, sheet=GSHEETS_URL_PROJ):
    clear = f"""
            UPDATE "{sheet}"
            SET {column} = NULL
            WHERE {column} IS NOT NULL
            """
    connection.execute(clear)
    
def batch_store_sheet(connection, entries, sheet=GSHEETS_URL_PROJ):
    columns = list(entries[0].keys())
    values = []
    for entry in entries:
        entry_str = '("' + '", "'.join([entry[col] for col in columns]) + '")'
        values.append(entry_str)
        
    for batch in range(0, len(values), GSHEET_API_CALLS_PM):
        values_str = ",\n".join(values[batch : min(batch + GSHEET_API_CALLS_PM, len(values))])
        insert = f''' 
                INSERT INTO "{sheet}" ({", ".join(columns)})
                VALUES {values_str}
                '''
        
        # Rotate through available connections
        connection_used = st.session_state.get("connection_used", 0)
        connections[connection_used].execute(insert)
        st.session_state.connection_used = (connection_used + 1) % len(connections)
        print(f"{min(batch + GSHEET_API_CALLS_PM, len(values))}/{len(values)} entries stored...")
        
        # May not be necessary once multiple users are used
        time.sleep(60)
