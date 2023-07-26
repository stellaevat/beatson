import streamlit as st
import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
from Bio import Entrez
from collections import defaultdict
import streamlit.components.v1 as components
from shillelagh.backends.apsw.db import connect
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from dataset import retrieve_projects
from active_learning import get_predictions

st.set_page_config(page_title="BioProject Annotation")

GSHEET_URL_PROJ = st.secrets["private_gsheets_url_proj"]
GSHEET_URL_PUB = st.secrets["private_gsheets_url_pub"]

Entrez.email = "stell.aeva@hotmail.com"
PROJECT_DB = "bioproject"
BASE_URL_PROJ = "https://www.ncbi.nlm.nih.gov/" + PROJECT_DB + "/"
BASE_URL_PUB = "https://pubmed.ncbi.nlm.nih.gov/"

DELIMITER = ", "
PLACEHOLDER = "-"
IDTYPE = "acc"
RETMAX = 10
RESULTS_PER_PAGE = 10
PROJECT_THRESHOLD = 10
LABEL_THRESHOLD = 3

tab_names = ["Annotate", "Search", "Predict"]
TAB_ANNOTATE, TAB_SEARCH, TAB_PREDICT = tab_names

project_columns = ["UID", "Accession", "Title", "Name", "Description", "Data_Type", "Scope", "Organism", "PMIDs", "Annotation", "Prediction", "To_Annotate"]
UID_COL, ACC_COL, TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL, ANNOT_COL, PREDICT_COL, LEARN_COL = project_columns

pub_columns = ["PMID", "Title", "Abstract", "MeSH", "Keywords"]
PMID_COL, PUBTITLE_COL, ABSTRACT_COL, MESH_COL, KEY_COL = pub_columns

annot_columns = [ACC_COL, TITLE_COL, ANNOT_COL]
search_columns = [ACC_COL, TITLE_COL]
predict_columns = [ACC_COL, TITLE_COL, PREDICT_COL]
detail_columns = [ACC_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL]
text_columns = [TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL]

search_msg = "Getting search results..."
loading_msg = "Loading project data..."

reload_btn = "↻"
next_btn = "Next"
prev_btn = "Previous"

primary_colour = "#81b1cc"
aggrid_css = {
    "#gridToolBar": {"display": "none;"},
    ".ag-theme-alpine, .ag-theme-alpine-dark": {"--ag-font-size": "12px;"},
    ".ag-cell": {"padding": "0px 12px;"},
}

markdown_translation = str.maketrans({char : '\\' + char for char in r'\`*_{}[]()#+-.!:><&'})

css = f"""
    <style>
        h3 {{font-size: 1.5rem; color: { primary_colour };}}
        thead {{display : none;}}
        th {{color: { primary_colour };}}
        [data-testid="stForm"] {{border: 0px; padding: 0px;}}
        button[kind="secondary"], button[kind="secondaryFormSubmit"] {{float: right; z-index: 1;}}
    </style>
"""
st.markdown(css, unsafe_allow_html=True)
    
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
def load_data(gsheet_url, _columns):
    query = f'SELECT * FROM "{gsheet_url}"'
    executed_query = connection.execute(query)
    df = pd.DataFrame(executed_query.fetchall())
    if not df.empty:
        df.columns = _columns
    return df
    
@st.cache_data(show_spinner=False)
def esearch(database, terms):
    if not terms:
        return None
    handle = Entrez.esearch(db=database, term=terms, retmax=RETMAX, idtype=IDTYPE)
    ids = Entrez.read(handle)["IdList"]
    return ids

 
@st.cache_resource(show_spinner=search_msg)
def api_search(search_terms):
    search_df, search_pub_df = None, None
    search_terms = [term.strip().lower() for term in search_terms.split() if term.strip()]
    
    # TODO: find synonyms
    
    ids = esearch(PROJECT_DB, "+OR+".join(search_terms))
    found = True if ids else False
    
    # Check both uid/acc columns because esearch unreliable
    ids_to_fetch = [project_id for project_id in ids if project_id not in project_df[[UID_COL, ACC_COL]].values]

    all_project_data, all_pub_data = retrieve_projects(ids_to_fetch)
    if all_project_data:
        search_df = pd.DataFrame(all_project_data)
        search_df.columns = project_columns
        if all_pub_data:
            search_pub_df = pd.DataFrame(all_pub_data)
            search_pub_df.columns = pub_columns
                
    return search_df, search_pub_df, found
            
            
@st.cache_resource(show_spinner=search_msg)
def local_search(search_terms, df):
    search_terms = [term.strip() for term in search_terms.split() if term.strip()]
    search_expr = r"(\b(" + "|".join(search_terms) + r")\b)"
    
    # TODO: find synonyms
    
    raw_counts = np.column_stack([df.astype(str)[col].str.count(search_expr, flags=re.IGNORECASE) for col in text_columns])
    total_counts = np.sum(raw_counts, axis=1)
    mask = np.where(total_counts > 0, True, False)
    search_df = df.loc[mask]
    
    search_df = search_df.sort_index(axis=0, key=lambda col: col.map(lambda i: total_counts[i]), ascending=False, ignore_index=True)
    return search_df
    
def display_search_feature(tab):
    search_terms = st.text_input("Search", label_visibility="collapsed", placeholder="Search", key=(tab + "_search")).strip()
    st.write("")
    
    if st.session_state.get(tab + "_prev_search", "") != search_terms:
        st.session_state[tab + "_selected_row_index"] = 0
        st.session_state[tab + "_selected_projects"] = []
    st.session_state[tab + "_prev_search"] = search_terms
    
    return search_terms

def id_to_url(base_url, page_id):
    return f'<a target="_blank" href="{base_url + str(page_id) + "/"}">{page_id}</a>'
 
def show_details(tab):
    st.session_state[tab + "_project_details_hidden"] = False
    
def hide_details(tab):
    st.session_state[tab + "_project_details_hidden"] = True
    
def go_to_next(tab, selected_row_index):
    st.session_state[tab + "_selected_row_index"] = selected_row_index + 1
    
def go_to_previous(tab, selected_row_index):
    st.session_state[tab + "_selected_row_index"] = selected_row_index - 1
    
def display_project_details(project):
    project = pd.Series({k : v.translate(markdown_translation) if v else v for (k, v) in project.to_dict().items()})
    
    st.write("")
    st.subheader(f"{project[TITLE_COL] if project[TITLE_COL] else project[NAME_COL] if project[NAME_COL] else project[ACC_COL]}")
    
    df = pd.DataFrame(project[detail_columns])
    df.loc[ACC_COL] = id_to_url(BASE_URL_PROJ, project[ACC_COL])
    
    if project[PUB_COL]:
        df.loc[PUB_COL] = DELIMITER.join([id_to_url(BASE_URL_PUB, pub_id) for pub_id in project[PUB_COL].split(DELIMITER)])
    
    for field in detail_columns:
        if not project[field]:
            df = df.drop(field, axis=0)
            
    st.write(df.to_html(render_links=True, escape=False), unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    if project[DESCR_COL]:
        st.write(project[DESCR_COL])
        
       
def display_navigation_buttons(tab, total_projects):
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    
    col1, col2 = st.columns(2)
    with col1:
        if selected_row_index > 0:
            st.button(prev_btn, on_click=go_to_previous, args=(tab, selected_row_index), key=(tab + "_prev"))
            
    with col2:
        if selected_row_index < total_projects - 1:
            st.button(next_btn, on_click=go_to_next, args=(tab, selected_row_index), key=(tab + "_next"))
            
    st.write("")
    st.write("")

@st.cache_data(show_spinner=False)
def get_grid_options(df, columns, starting_page, selected_row_index, selection_mode="single"):
    options_dict = {
        "enableCellTextSelection" : True,
        
        "onFirstDataRendered" : JsCode(f"""
            function onFirstDataRendered(params) {{
                params.api.paginationGoToPage({ starting_page });
            }}
        """),
        
        "onPaginationChanged" : JsCode(f"""
            function onPaginationChanged(params) {{
                let doc = window.parent.document;
                let buttons = Array.from(doc.querySelectorAll('button[kind="secondary"]'));
                let reloadButton = buttons.find(el => el.innerText === "{ reload_btn }");
                
                if (typeof reloadButton !== "undefined") {{
                    reloadButton.click();
                }}
            }}
        """),
    }
    
    if selection_mode == "single":
        options_dict["getRowStyle"] = JsCode(f"""
            function(params) {{
                if (params.rowIndex == { str(selected_row_index) }) {{
                    return {{'background-color': '{ primary_colour }', 'color': 'black'}};
                }}
            }}
        """)
    
    builder = GridOptionsBuilder.from_dataframe(df[columns])
    
    builder.configure_column(ACC_COL, lockPosition="left", suppressMovable=True, width=110)
    builder.configure_column(TITLE_COL, flex=3)
    builder.configure_column(columns[-1], flex=1)
    builder.configure_selection(selection_mode=selection_mode)
    builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=RESULTS_PER_PAGE)
    builder.configure_grid_options(**options_dict)

    grid_options = builder.build()
    return grid_options
    
def display_interactive_grid(tab, df, columns, nav_buttons=True, selection_mode="single"):
    rerun = st.session_state.get(tab + "_rerun", 0)
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    starting_page = selected_row_index // RESULTS_PER_PAGE

    grid_options = get_grid_options(df, columns, starting_page, selected_row_index, selection_mode)
    grid = AgGrid(
        df[columns].replace('', None).fillna(value=PLACEHOLDER), 
        gridOptions=grid_options,
        width="100%",
        theme="alpine",
        update_mode=(GridUpdateMode.SELECTION_CHANGED if selection_mode=="single" else GridUpdateMode.NO_UPDATE),
        custom_css=aggrid_css,
        allow_unsafe_jscode=True,
        reload_data=False,
        enable_enterprise_modules=False
    )
    selected_row = grid['selected_rows']
    selected_df = pd.DataFrame(selected_row)
    previous_page = st.session_state.get(tab + "_starting_page", 0)
    project_details_hidden = st.session_state.get(tab + "_project_details_hidden", True)
    
    if project_details_hidden:
        st.button("Show details", key=(tab + "_show"), on_click=show_details, args=(tab,))
    else:
        st.button("Hide details", key=(tab + "_hide"), on_click=hide_details, args=(tab,))

    if rerun:
        if not project_details_hidden:
            display_project_details(df.iloc[selected_row_index])
            if nav_buttons:
                display_navigation_buttons(tab, len(df))
            
        st.session_state[tab + "_starting_page"] = starting_page
        st.session_state[tab + "_rerun"] = 0

    elif not selected_df.empty:
        selected_mask = df[ACC_COL].isin(selected_df[ACC_COL])
        selected_data = df.loc[selected_mask]
        
        selected_row_index = selected_data.index.tolist()[0]
        st.session_state[tab + "_selected_row_index"] = selected_row_index
        
        selected_id = df.iloc[selected_row_index][ACC_COL]
        selected_projects = st.session_state.get(tab + "_selected_projects", [])
        if selected_id not in selected_projects:
            st.session_state[tab + "_selected_projects"] = selected_projects + [selected_id]
            
        st.session_state[tab + "_rerun"] = 1
        
        # Rerun to have selection displayed
        st.experimental_rerun()   
        
    elif not project_details_hidden:
        display_project_details(df.iloc[selected_row_index])
        if nav_buttons:
            display_navigation_buttons(tab, len(df))


def update_sheet(project_id, value, column, sheet=GSHEET_URL_PROJ):
    update = f"""
        UPDATE "{sheet}"
        SET {column} = "{value}"
        WHERE {ACC_COL} = "{project_id}"
    """
    connection.execute(update)
    
def insert_sheet(values, columns=project_columns, sheet=GSHEET_URL_PROJ):
    values_str = '("' + '", "'.join([str(val) for val in values]) + '")'
    insert = f"""
        INSERT INTO "{sheet}" ({", ".join(columns)})
        VALUES {values_str}
    """
    connection.execute(insert)
   
def clear_sheet_column(column, sheet=GSHEET_URL_PROJ):
    clear = f"""
            UPDATE "{sheet}"
            SET {column} = NULL
            WHERE {column} IS NOT NULL
            """
    connection.execute(clear)

 
def add_to_dataset(tab, df, new_pub_df, project_ids=None):
    if project_ids:
        df = df[df[ACC_COL].isin(project_ids)]
        
    for i, project in pd.DataFrame(df).iterrows():
        if project[ACC_COL] not in project_df[ACC_COL].unique():
            project_df.loc[len(project_df.index)] = project.tolist()
            insert_sheet(project.fillna(value='').tolist())
            
            publications = project[PUB_COL]
            if publications and new_pub_df is not None:
                for pmid in publications.split(DELIMITER):
                    if pmid in new_pub_df[PMID_COL].unique() and pmid not in pub_df[PMID_COL].unique():
                        pub_values = new_pub_df.loc[new_pub_df[PMID_COL] == pmid].squeeze()
                        insert_sheet(pub_values.tolist(), pub_columns, GSHEET_URL_PUB)
                        pub_df.loc[len(pub_df)] = pub_values
                        
            # Display update
            api_project_df.drop(i, axis=0, inplace=True)
            selected_projects = st.session_state.get(tab + "_selected_projects", [])
            if project[ACC_COL] in selected_projects:
                selected_projects.remove(project[ACC_COL])
                st.session_state[tab + "_selected_projects"] = selected_projects
                
    st.session_state[tab + "_selected_row_index"] = 0
                        
                    
def display_add_to_dataset_feature(tab, df, new_pub_df):
    st.header("Add to dataset")
    col1, col2, col3 = st.columns(3)
    
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = df.iloc[selected_row_index][ACC_COL]
    selected_projects = st.session_state.get(tab + "_selected_projects", [])
    
    with st.form(key=(tab + "_add_form")):
        add_selection = st.multiselect("Add selection", df[ACC_COL], default=selected_projects, label_visibility="collapsed", key=(tab + "_add_selection"))
        
        # Buttons as close as possible without line-breaks
        col1, col2 = st.columns([0.818, 0.182])
        with col1:
            st.form_submit_button("Add selection", on_click=add_to_dataset, args=(tab, df, new_pub_df, add_selection))
        with col2:
            st.form_submit_button("Add all results", on_click=add_to_dataset, args=(tab, df, new_pub_df, df[ACC_COL].tolist()))
        


def get_project_labels(project_id):
    if project_id in project_df[ACC_COL].unique():
        project_labels = project_df[project_df[ACC_COL] == project_id][ANNOT_COL].item()
        if project_labels:
            return project_labels.split(DELIMITER)
    return []
    

def update_labels(tab, df, new_pub_df=None):
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = df.iloc[selected_row_index][ACC_COL]
    
    existing = st.session_state.get(tab + "_labels", "")
    new = st.session_state.get(tab + "_new", "")
    updated_labels = (set(existing) | {n.strip() for n in new.split(",")}) - {""}
    updated_labels_str = DELIMITER.join(sorted(list(updated_labels)))
    
    if project_id in project_df[ACC_COL].unique():
        original_labels = get_project_labels(project_id)
        if updated_labels ^ set(original_labels):
            update_sheet(project_id, updated_labels_str, ANNOT_COL)
            
            if updated_labels_str:
                project_df.loc[project_df[ACC_COL] == project_id, ANNOT_COL] = updated_labels_str
                
                # So that selected row still within bounds
                if project_df.loc[project_df[ACC_COL] == project_id, LEARN_COL].item() == True:
                    st.session_state[TAB_PREDICT + "_selected_row_index"] = 0
            else:
                project_df.loc[project_df[ACC_COL] == project_id, ANNOT_COL] = None
    else:
        add_to_dataset(df.iloc[selected_row_index], new_pub_df)
            
    st.session_state[tab + "_new"] = ""
    

@st.cache_data(show_spinner=False)    
def get_label_options(project_df):
    label_options = set()
    if not project_df.empty:
        for annotation in project_df[ANNOT_COL]:
            if annotation is not None:
                label_options.update(set(annotation.split(DELIMITER)))
        label_options = sorted(list(label_options))
    return label_options
    
    
def display_annotation_feature(tab, df, new_pub_df=None, allow_new=True):
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    project_id = df.iloc[selected_row_index][ACC_COL]
    
    label_options = get_label_options(project_df)
    original_labels = get_project_labels(project_id)
    st.session_state[tab + "_labels"] = original_labels

    with st.form(key=(tab + "_annotation_form")):
        st.write(f"Edit **{project_id}** labels:")
        if label_options and allow_new:
            col1, col2 = st.columns(2)
            with col1:
                labels = st.multiselect("Choose", label_options, label_visibility="collapsed", key=(tab + "_labels"))
            with col2:
                new = st.text_input("Create new", placeholder="Or create new (comma-separated)", label_visibility="collapsed", key=(tab + "_new"))
        elif label_options:
            labels = st.multiselect("Choose", label_options, label_visibility="collapsed", key=(tab + "_labels"))
        else:
            labels = ""
            new = st.text_input("Create new", placeholder="Create new (comma-separated)", label_visibility="collapsed", key=(tab + "_new"))

        st.form_submit_button("Update", on_click=update_labels, args=(tab, df, new_pub_df))
        
        
@st.cache_resource(show_spinner=False)
def get_label_matrix(df):
    labels = set()
    for annotation in df[ANNOT_COL]:
        if annotation is not None:
            labels.update(set(annotation.split(DELIMITER)))
    
    # Sort labels
    i = 0
    label_to_index = {}
    index_to_label = {}
    for label in labels:
        label_to_index[label] = i
        index_to_label[i] = label
        i += 1
    
    y_labelled = np.zeros((len(df), len(labels)))
    for i, annotation in enumerate(df[ANNOT_COL]):
        if annotation is not None:
            for label in annotation.split(DELIMITER):
                y_labelled[i, label_to_index[label]] = 1
            
    return y_labelled, label_to_index, index_to_label
 
@st.cache_resource(show_spinner=False) 
def get_sample_matrix(df, pub_df):
    X_labelled = []
    for i, project in df.iterrows():
        text = " ".join([field for field in project[text_columns] if field is not None])
        if project[PUB_COL]:
            for pmid in project[PUB_COL].split(DELIMITER):
                text += " " + " ".join([field for field in pub_df.loc[pub_df[PMID_COL] == pmid].iloc[0] if field is not None])
        X_labelled.append(text)
    return X_labelled
 
@st.cache_data(show_spinner="Checking dataset...") 
def check_dataset(project_df):
    unlabelled_df = project_df[project_df[ANNOT_COL].isnull()]
    labelled_df = project_df[project_df[ANNOT_COL].notnull()]
    y_labelled, label_to_index, index_to_label = get_label_matrix(labelled_df)
    
    label_sums = np.sum(y_labelled, axis=0)
    project_sum = np.sum(y_labelled, axis=1)
    labelled_projects = len(project_sum[project_sum > 0]) 
    rare_labels = (label_sums < LABEL_THRESHOLD).nonzero()[0]

    if labelled_projects == len(project_df):
        message.write("All projects in the dataset have been manually annotated. Use the **Search** tab to find and add unannotated projects.")
    elif labelled_projects < PROJECT_THRESHOLD:
        message.write(f"So far **{labelled_projects} projects** have been annotated. For the algorithm to work well please find at least **{PROJECT_THRESHOLD - labelled_projects}** more project{'s' if PROJECT_THRESHOLD - labelled_projects > 1 else ''} to annotate.") 
    elif len(rare_labels) > 0:
        message.write(f"Some labels have less than **{LABEL_THRESHOLD} samples**. For the algorithm to work well please find more projects to label as: **{', '.join([index_to_label[i] for i in rare_labels])}**.")
    else:
        X_labelled = get_sample_matrix(labelled_df, pub_df)
        X_unlabelled = get_sample_matrix(unlabelled_df, pub_df)
        labels = sorted(list(label_to_index.keys()), key=lambda x: label_to_index[x])
        return X_labelled, X_unlabelled, y_labelled, labels
        
    return None, None, None, None

@st.cache_data(show_spinner=False) 
def int_column(col):
    return pd.Series([int(val) if (val and val.isnumeric()) else 0 for val in col])
    
@st.cache_data(show_spinner="Processing predictions...")
def process_predictions(y_predicted, y_probabilities, to_annotate, labels, df):
    labels = np.array(labels)
    unlabelled_df = project_df[project_df[ANNOT_COL].isnull()]
    
    count = 0
    for i, project_id in enumerate(unlabelled_df[ACC_COL]):
        predicted_mask = np.where(y_predicted[i] > 0, True, False)
        predicted_str = DELIMITER.join(sorted(labels[predicted_mask]))
        if predicted_str != project_df.loc[project_df[ACC_COL] == project_id, PREDICT_COL].item():
            count += 1
            update_sheet(project_id, predicted_str, PREDICT_COL)
            project_df.loc[project_df[ACC_COL] == project_id, PREDICT_COL] = predicted_str
            
    print("Total: ", len(unlabelled_df))
    print("Updated: ", count)
        
    if to_annotate:
        clear_sheet_column(LEARN_COL)
        project_df[LEARN_COL] = None
        
        learn_df = unlabelled_df.iloc[to_annotate, :]
        for i, project_id in enumerate(learn_df[ACC_COL]):
            update_sheet(project_id, str(i+1), LEARN_COL)
            project_df.loc[project_df[ACC_COL] == project_id, LEARN_COL] = str(i+1)
            
    
st.button(reload_btn, key="reload_btn") 
st.title("BioProject Annotation")
annotate, search, predict = st.tabs(tab_names)

connection = connect_gsheets_api()
project_df = load_data(GSHEET_URL_PROJ, project_columns)
pub_df = load_data(GSHEET_URL_PUB, pub_columns)
    
with annotate:
    st.header("Annotate projects")
    annotate_df = project_df

    if not project_df.empty:
        find_terms = display_search_feature(TAB_ANNOTATE)
        
        if find_terms:   
            search_df = local_search(find_terms, project_df)
            if search_df is not None and not search_df.empty:
                st.write(f"Results for '{find_terms}':")
                annotate_df = search_df
            else:
                st.write(f"No results for '{find_terms}'. All projects:")
        
        if not annotate_df.empty:
            display_interactive_grid(TAB_ANNOTATE, annotate_df, annot_columns)
            display_annotation_feature(TAB_ANNOTATE, annotate_df)
        
    else:
        st.write("Annotation dataset unavailable. Use the Search tab to search the BioProject database directly.")
    

with search:
    st.header("Search BioProject")
    
    api_terms = display_search_feature(TAB_SEARCH)
    if api_terms:
        api_project_df, api_pub_df, found = api_search(api_terms)
        if api_project_df is not None and not api_project_df.empty:
            st.write(f"Results for '{api_terms}':")
            display_interactive_grid(TAB_SEARCH, api_project_df, search_columns)
            display_add_to_dataset_feature(TAB_SEARCH, api_project_df, api_pub_df)
        elif found:
            st.write(f"No results for '{api_terms}' which are not already in the dataset. Try looking for something else.")
        else:
            st.write(f"No results for '{api_terms}'. Check for typos or try looking for something else.")
  
with predict:
    st.header("Predict annotations")
    
    message = st.empty()
    message.write("Click **Start** to get predictions for all unannotated projects.")
    start_button = st.button("Start", key="start_button")

    if start_button:
        X_labelled, X_unlabelled, y_labelled, labels = check_dataset(project_df)
        if X_labelled:
            y_predicted, y_probabilities, to_annotate, f1_micro_ci, f1_macro_ci = get_predictions(X_labelled, y_labelled, X_unlabelled)
            st.session_state.f1_micro_ci = f1_micro_ci
            st.session_state.f1_macro_ci = f1_macro_ci
            
            # Columns irrelevant to method cacheing dropped
            df = project_df.drop([PREDICT_COL, LEARN_COL], axis=1)
            process_predictions(y_predicted, y_probabilities, to_annotate, labels, df)
            
            st.session_state.new_predictions = True
    
    predict_df = project_df[project_df[PREDICT_COL].notnull()]
    if predict_df is not None and not predict_df.empty:
        if st.session_state.get("new_predictions", False):
            st.header("Predicted labels")
            
            f1_micro_ci = st.session_state.f1_micro_ci
            st.write(f"Micro-f1: {np.mean(f1_micro_ci):.3f}, with 95% confidence interval ({f1_micro_ci[0]:.3f}, {f1_micro_ci[1]:.3f})")
            
            f1_macro_ci = st.session_state.f1_macro_ci
            st.write(f"Macro-f1: {np.mean(f1_macro_ci):.3f}, with 95% confidence interval ({f1_macro_ci[0]:.3f}, {f1_macro_ci[1]:.3f})")
        else:
            st.header("Previously predicted labels")
        display_interactive_grid(TAB_PREDICT, predict_df, predict_columns, nav_buttons=False)
    
    learn_section_name = "Learn"
    learn_df = project_df[int_column(project_df[LEARN_COL]) > 0]    
    if learn_df is not None and not learn_df.empty:
        st.header("Improve predictions")
        st.write("To improve performance, consider annotating the following projects:")
        # Sort by annotation importance to active learning
        learn_df = learn_df.sort_values(LEARN_COL, axis=0, ignore_index=True, key=lambda col: int_column(col))
        display_interactive_grid(learn_section_name, learn_df, annot_columns)
        display_annotation_feature(learn_section_name, learn_df)
        
import streamlit.components.v1 as components

components.html(
    f"""
        <script>
            function addTabReruns () {{
                let doc = window.parent.document;
                let buttons = Array.from(doc.querySelectorAll('button[kind="secondary"]'));
                let reloadButton = buttons.find(el => el.innerText === "{ reload_btn }");

                let tabs = doc.querySelectorAll('button[role="tab"]');
                
                for (let i = 0; i < tabs.length; i++) {{
                    tabs[i].addEventListener("click", function () {{
                        reloadButton.click();
                    }});
                }}
            }}
            
            function positionPreviousButton () {{
                let doc = window.parent.document;
                let buttons = Array.from(doc.querySelectorAll('button[kind="secondary"]'));
                let previousButtons = buttons.filter(el => el.innerText === "{ prev_btn }");
                
                for (let i = 0; i < previousButtons.length; i++) {{
                    previousButtons[i].style.float = "left";
                }}
            }} 
            
            window.onload = addTabReruns;
            setInterval(positionPreviousButton, 500);
        </script>
    """,
    height=0, width=0
)