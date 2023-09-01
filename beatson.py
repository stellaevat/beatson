import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import date
import streamlit.components.v1 as components
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from gsheets import get_gsheets_urls, get_gsheets_columns, get_delimiter, connect_gsheets_api, load_sheet, update_sheet, insert_sheet, clear_sheet_column
from search import api_search, local_search, display_search_feature
from active_learning import get_predictions

st.set_page_config(page_title="BioProject Annotation")

GSHEETS_URL_PROJ, GSHEETS_URL_PUB, GSHEETS_URL_METRICS = get_gsheets_urls()
DELIMITER = get_delimiter()

PLACEHOLDER = "-"
RESULTS_PER_PAGE = 10
PROJECT_THRESHOLD = 10
ANNOT_SUGGESTIONS = 10
LABEL_THRESHOLD = 3
CONFIDENCE_THRESHOLD = 0.75

tab_names = ["Annotate", "Search", "Predict"]
TAB_ANNOTATE, TAB_SEARCH, TAB_PREDICT = tab_names

project_columns, pub_columns, metric_columns = get_gsheets_columns()
UID_COL, ACC_COL, TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL, ANNOT_COL, PREDICT_COL, SCORE_COL, LEARN_COL  = project_columns
PMID_COL, PUBTITLE_COL, ABSTRACT_COL, MESH_COL, KEY_COL = pub_columns

annot_columns = [ACC_COL, TITLE_COL, ANNOT_COL, PREDICT_COL]
search_columns = [ACC_COL, TITLE_COL]
predict_columns = [ACC_COL, TITLE_COL, ANNOT_COL, PREDICT_COL]
detail_columns = [ACC_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL]
text_columns = [TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL]
export_columns = [UID_COL, ACC_COL, TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL, ANNOT_COL, PREDICT_COL]

reload_btn = "↻"
export_btn = "Export to CSV"
show_btn = "Show details"
hide_btn = "Hide details"
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
        th {{color: { primary_colour }; font-weight: normal;}}
        [data-testid="stForm"] {{border: 0px; padding: 0px;}}
        button[kind="secondary"], button[kind="secondaryFormSubmit"] {{float: right; z-index: 1;}}
    </style>
"""
st.markdown(css, unsafe_allow_html=True)
    


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
    st.write("")
    
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
        st.write(f"**{DESCR_COL}:** {project[DESCR_COL]}")
        
    labels = ""
    if project[ANNOT_COL]:
        labels += f"""**Manual annotation:** *{project[ANNOT_COL]}*  
        """ 
    if project[PREDICT_COL]:
        labels += f"""**Predicted annotation:** *{project[PREDICT_COL]}* ({project[SCORE_COL]} confidence)  
        """
    if labels:
        st.write(labels)
        
    st.write("")
        
       
def display_navigation_buttons(tab, total_projects):
    selected_row_index = st.session_state.get(tab + "_selected_row_index", 0)
    
    col1, col2 = st.columns(2)
    with col1:
        if selected_row_index > 0:
            st.button(prev_btn, on_click=go_to_previous, args=(tab, selected_row_index), key=(tab + "_prev"))
            
    with col2:
        if selected_row_index < total_projects - 1:
            st.button(next_btn, on_click=go_to_next, args=(tab, selected_row_index), key=(tab + "_next"))



@st.cache_data(show_spinner=False)
def get_grid_options(df, columns, starting_page, selected_row_index, selection_mode="single"):
    options_dict = {
        "enableCellTextSelection" : True,
        "enableBrowserTooltips" : True,
        
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
        
    tooltip_cell = JsCode(""" 
        class TooltipCellRenderer {
            init(params) {
                this.eGui = document.createElement('span');
                this.eGui.innerHTML = '<span title="' + params.value + '">' + params.value + '</span>';
            }
          
            getGui() {
                return this.eGui;
            }
        }
    """)
    
    builder = GridOptionsBuilder.from_dataframe(df[columns])
    
    builder.configure_default_column(cellRenderer=tooltip_cell)
    if ACC_COL in columns:
        builder.configure_column(ACC_COL, lockPosition="left", suppressMovable=True, width=115)
    if TITLE_COL in columns:
        builder.configure_column(TITLE_COL, flex=2)
    if ANNOT_COL in columns:
        builder.configure_column(ANNOT_COL, flex=1)
    if PREDICT_COL in columns:
        builder.configure_column(PREDICT_COL, flex=1)
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
    
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Export to CSV", df[export_columns].rename(columns={ANNOT_COL:"Manual_Annotation", PREDICT_COL: "Predicted_Annotation"}).to_csv(index=False).encode('utf-8'), "BioProjct_Annotation.csv", "text/csv", key=(tab + "_export"))
    with col2:
        if project_details_hidden:
            st.button(show_btn, key=(tab + "_show"), on_click=show_details, args=(tab,))
        else:
            st.button(hide_btn, key=(tab + "_hide"), on_click=hide_details, args=(tab,))
        

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
            
    st.write("")




 
def add_to_dataset(tab, df, new_pub_df):
    for i, project in pd.DataFrame(df).iterrows():
        if project[ACC_COL] not in project_df[ACC_COL].unique():
            project_df.loc[len(project_df.index)] = project.tolist()
            insert_sheet(connection, project.fillna(value='').tolist())
            
            publications = project[PUB_COL]
            if publications and new_pub_df is not None:
                for pmid in publications.split(DELIMITER):
                    if pmid in new_pub_df[PMID_COL].unique() and pmid not in pub_df[PMID_COL].unique():
                        pub_values = new_pub_df.loc[new_pub_df[PMID_COL] == pmid].squeeze()
                        insert_sheet(connection, pub_values.tolist(), pub_columns, GSHEETS_URL_PUB)
                        pub_df.loc[len(pub_df)] = pub_values
                        
            # Display update
            api_df.drop(i, axis=0, inplace=True)
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
            st.form_submit_button("Add selection", on_click=add_to_dataset, args=(tab, df[df[ACC_COL].isin(add_selection)], new_pub_df))
        with col2:
            st.form_submit_button("Add all results", on_click=add_to_dataset, args=(tab, df, new_pub_df))
        


def get_project_labels(project_id, column=ANNOT_COL):
    if project_id in project_df[ACC_COL].unique():
        project_labels = project_df[project_df[ACC_COL] == project_id][column].item()
            
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
            update_sheet(connection, project_id, {ANNOT_COL : updated_labels_str})
            
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
        st.write(f"Edit **{project_id}** annotation:")
        if label_options and allow_new:
            col1, col2 = st.columns(2)
            with col1:
                labels = st.multiselect("Choose", label_options, label_visibility="collapsed", key=(tab + "_labels"))
            with col2:
                new = st.text_input("Create new", placeholder="Or create new (comma-separated)", label_visibility="collapsed", autocomplete="off", key=(tab + "_new"))
        elif label_options:
            labels = st.multiselect("Choose", label_options, label_visibility="collapsed", key=(tab + "_labels"))
        else:
            labels = ""
            new = st.text_input("Create new", placeholder="Create new (comma-separated)", label_visibility="collapsed", autocomplete="off", key=(tab + "_new"))

        st.form_submit_button("Update", on_click=update_labels, args=(tab, df, new_pub_df))
        
        
@st.cache_resource(show_spinner=False)
def get_label_matrix(df):
    labels = set()
    for annotation in df[ANNOT_COL]:
        if annotation is not None:
            labels.update(set(annotation.split(DELIMITER)))
    labels = sorted(list(labels))

    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    
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
    
@st.cache_data(show_spinner=False) 
def float_column(col):
    return pd.Series([float(val) if (val and val.isnumeric()) else 0 for val in col])
    
@st.cache_data(show_spinner="Processing predictions...")
def process_predictions(y_predicted, y_scores, labels, df):
    labels = np.array(labels)
    to_annotate = np.argsort(y_scores)[:ANNOT_SUGGESTIONS].tolist() if len(y_scores) > ANNOT_SUGGESTIONS else np.argsort(y_scores).tolist()
    unlabelled_df = project_df[project_df[ANNOT_COL].isnull()]
    
    for i, project_id in enumerate(unlabelled_df[ACC_COL]):
        predicted_mask = np.where(y_predicted[i] > 0, True, False)
        predicted_str = DELIMITER.join(sorted(labels[predicted_mask]))
        score = "%.3f" % y_scores[i]

        old_prediction = project_df.loc[project_df[ACC_COL] == project_id, PREDICT_COL].item()
        old_prediction = old_prediction if old_prediction else ""
        old_score = project_df.loc[project_df[ACC_COL] == project_id, SCORE_COL].item()
        old_score = old_score if old_score else ""
        
        if predicted_str != old_prediction or score != old_score:
            update_sheet(connection, project_id, {PREDICT_COL : predicted_str, SCORE_COL : score})
            project_df.loc[project_df[ACC_COL] == project_id, PREDICT_COL] = predicted_str
            project_df.loc[project_df[ACC_COL] == project_id, SCORE_COL] = score
        
    if to_annotate:
        old_learn_df = project_df[int_column(project_df[LEARN_COL]) > 0]
        new_learn_df = unlabelled_df.iloc[to_annotate, :].reset_index(drop=True)
        
        updates = {project[ACC_COL] : None for (i, project) in old_learn_df.iterrows() if project[ACC_COL] != new_learn_df[new_learn_df[LEARN_COL] == project[LEARN_COL]][ACC_COL].item()}
        for (i, project) in new_learn_df.iterrows():
            if project[ACC_COL] != old_learn_df[old_learn_df[LEARN_COL] == project[LEARN_COL]][ACC_COL].item():
                updates[project[ACC_COL]] = str(i+1)
          
        # Update with minimum API calls  
        if updates:
            if len(updates) < len(new_learn_df) + 1:
                for (project_id, order) in updates.items():
                    update_sheet(connection, project_id, {LEARN_COL : order})
                    project_df.loc[project_df[ACC_COL] == project_id, LEARN_COL] = order
            else:
                clear_sheet_column(connection, LEARN_COL)
                project_df[LEARN_COL] = None
                for (project_id, order) in updates.items():
                    if order is not None:
                        update_sheet(connection, project_id, {LEARN_COL : order})
                        project_df.loc[project_df[ACC_COL] == project_id, LEARN_COL] = order
    
    
    # Clear predictions from earlier runs (labelled projects not updated above)
    labelled_df = project_df[project_df[ANNOT_COL].notnull()]
    for i, project_id in enumerate(labelled_df[ACC_COL]):
        if project_df.loc[project_df[ACC_COL] == project_id, SCORE_COL].item():
            update_sheet(connection, project_id, {PREDICT_COL : None, SCORE_COL : None})
            project_df.loc[project_df[ACC_COL] == project_id, [PREDICT_COL, SCORE_COL]] = None
            
    
st.button(reload_btn, key="reload_btn") 
st.title("BioProject Annotation")
annotate, search, predict = st.tabs(tab_names)

connection = connect_gsheets_api(0)
project_df = load_sheet(connection, project_columns, GSHEETS_URL_PROJ)
pub_df = load_sheet(connection, pub_columns, GSHEETS_URL_PUB)
metric_df = load_sheet(connection, metric_columns, GSHEETS_URL_METRICS)

with annotate:
    st.header("Annotate projects")
    annotate_df = project_df

    if not project_df.empty:
        find_terms = display_search_feature(TAB_ANNOTATE)
        
        if find_terms:   
            find_df = local_search(find_terms, project_df, text_columns)
            if find_df is not None and not find_df.empty:
                st.write(f"Results for '{find_terms}':")
                annotate_df = find_df
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
        api_df, api_pub_df, found = api_search(api_terms, project_df[[UID_COL, ACC_COL]].values)
        if api_df is not None and not api_df.empty:
            st.write(f"Results for '{api_terms}':")
            display_interactive_grid(TAB_SEARCH, api_df, search_columns)
            display_add_to_dataset_feature(TAB_SEARCH, api_df, api_pub_df)
        elif found:
            st.write(f"No results for '{api_terms}' that are not already in the dataset. Try looking for something else.")
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
            train_size = len(X_labelled)
            test_size = len(X_unlabelled)
            dataset_hash = hashlib.shake_256(pd.util.hash_pandas_object(project_df[[ACC_COL, ANNOT_COL]].sort_values(ACC_COL, axis=0), index=False).values).hexdigest(8)
            
            y_predicted, y_scores, f1_micro_ci, f1_macro_ci = get_predictions(X_labelled, X_unlabelled, y_labelled)

            st.session_state.f1_micro_ci = f1_micro_ci
            st.session_state.f1_macro_ci = f1_macro_ci
            
            # Columns irrelevant to method cacheing dropped
            df = project_df.drop([PREDICT_COL, LEARN_COL], axis=1)
            process_predictions(y_predicted, y_scores, labels, df)
            
            metric_row = np.array([date.today().strftime("%d/%m/%Y"), dataset_hash, train_size, test_size, np.mean(f1_micro_ci), np.mean(f1_macro_ci)])
            if metric_df.empty or not (metric_df == metric_row).all(1).any():
                insert_sheet(connection, metric_row, metric_columns, GSHEETS_URL_METRICS)
                metric_df.loc[len(metric_df)] = metric_row

            st.session_state.new_predictions = True
    
    predict_df = project_df[project_df[PREDICT_COL].notnull()]
    if not predict_df.empty:
        if st.session_state.get("new_predictions", False):
            st.header("Predicted")
            metrics = ""
            
            # How many above confidence threshold            
            confidence_pct = 100 * (float_column(predict_df[SCORE_COL]) > CONFIDENCE_THRESHOLD).sum() / len(predict_df)
            st.write(f"""**{confidence_pct:.0f}%** of all project annotations were predicted with a confidence score over {CONFIDENCE_THRESHOLD}  
            """)
            
            f1_micro_ci = st.session_state.f1_micro_ci
            metrics += f"""Micro-f1: **{np.mean(f1_micro_ci):.3f}**, with 95% CI ({f1_micro_ci[0]:.3f}, {f1_micro_ci[1]:.3f})  
            """
            
            f1_macro_ci = st.session_state.f1_macro_ci
            metrics += f"""Macro-f1: **{np.mean(f1_macro_ci):.3f}**, with 95% CI ({f1_macro_ci[0]:.3f}, {f1_macro_ci[1]:.3f})  
            """

            st.write(metrics)
            
            
            
            
            
        else:
            st.header("Previously predicted")
        display_interactive_grid(TAB_PREDICT, predict_df, predict_columns)
        display_annotation_feature(TAB_PREDICT, predict_df)
    
    learn_section_name = "Learn"
    learn_df = project_df[int_column(project_df[LEARN_COL]) > 0]    
    if not learn_df.empty:
        st.header("Improve prediction algorithm")
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
            
            function styleButtons () {{
                let doc = window.parent.document;
                let buttons = Array.from(doc.querySelectorAll('button[kind="secondary"]'));
                let exportButtons = buttons.filter(el => el.innerText === "{ export_btn }");
                let previousButtons = buttons.filter(el => el.innerText === "{ prev_btn }");
                
                for (let i = 0; i < exportButtons.length; i++) {{
                    exportButtons[i].style.float = "left";
                    exportButtons[i].style.backgroundColor = "#4c4c4c";
                    exportButtons[i].style.borderColor = "#4c4c4c";
                    exportButtons[i].style.color = "white";
                }}
                
                for (let i = 0; i < previousButtons.length; i++) {{
                    previousButtons[i].style.float = "left";
                }}  
            }} 
            
            window.onload = addTabReruns;
            setInterval(styleButtons, 500);
        </script>
    """,
    height=0, width=0
)