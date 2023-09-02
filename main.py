import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import date
import streamlit.components.v1 as components
from features.gsheets import connect_gsheets_api, load_sheet, insert_sheet, get_gsheets_urls, get_gsheets_columns
from features.search import display_search_feature, api_search, local_search
from features.grids import display_interactive_grid, get_grid_buttons, get_primary_colour
from features.annotate import display_annotation_feature, display_add_to_dataset_feature
from features.predict import process_dataset, process_predictions
from features.active_learning import active_learning

st.set_page_config(page_title="BioProject Annotation")

GSHEETS_URL_PROJ, GSHEETS_URL_PUB, GSHEETS_URL_METRICS = get_gsheets_urls()
project_columns, pub_columns, metric_columns = get_gsheets_columns()

UID_COL, ACC_COL, TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL, ANNOT_COL, PREDICT_COL, SCORE_COL, LEARN_COL  = project_columns
PMID_COL, PUBTITLE_COL, ABSTRACT_COL, MESH_COL, KEY_COL = pub_columns

annot_columns = [ACC_COL, TITLE_COL, ANNOT_COL, PREDICT_COL]
search_columns = [ACC_COL, TITLE_COL]
predict_columns = [ACC_COL, TITLE_COL, ANNOT_COL, PREDICT_COL]
text_columns = [TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL]

tab_names = ["Annotate", "Search", "Predict"]
TAB_ANNOTATE, TAB_SEARCH, TAB_PREDICT = tab_names
help_btn = "Help"

CONFIDENCE_THRESHOLD = 0.75

reload_btn, export_btn, show_btn, hide_btn, next_btn, prev_btn = get_grid_buttons()
primary_colour = get_primary_colour()

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
    
    
@st.cache_data(show_spinner=False) 
def int_column(col):
    return pd.Series([int(val) if (val and val.isnumeric()) else 0 for val in col])
    
    
@st.cache_data(show_spinner=False) 
def float_column(col):
    return pd.Series([float(val) if (val and val.isnumeric()) else 0 for val in col])
    
    
def toggle_help():
    hidden = st.session_state.get("help_hidden", True)
    if hidden:
        st.session_state["help_hidden"] = False
    else:
        st.session_state["help_hidden"] = True
    
    
# Header    
st.button(reload_btn, key="reload_btn") 
st.title("BioProject Annotation")
st.write("Annotate and predict annotations for NCBI BioProjects, with fields that serve your specific research goals.")
annotate, search, predict = st.tabs(tab_names)

# Load Google sheet data
connection = connect_gsheets_api()
project_df = load_sheet(connection, project_columns, GSHEETS_URL_PROJ)
pub_df = load_sheet(connection, pub_columns, GSHEETS_URL_PUB)
metric_df = load_sheet(connection, metric_columns, GSHEETS_URL_METRICS)

st.session_state.project_df = project_df
st.session_state.pub_df = pub_df


# Annotate tab
with annotate:
    annot_help = st.button(help_btn, key=(TAB_ANNOTATE + "_help"), on_click=toggle_help)
    if not st.session_state.get("help_hidden", True):
        st.info("""**Help (Annotate):**   
        •  Explore a subset of the BioProject database, to annotate it with your fields of interest.   
        •  Search, filter and view project details to discover relevant projects.   
        •  Create labels to annotate with, or choose from the drop-down list of the ones you have already used.""")
        
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
            display_annotation_feature(TAB_ANNOTATE, connection, annotate_df)
        
    else:
        st.write("Annotation dataset unavailable. Use the Search tab to search the BioProject database directly.")
    

# Search tab
with search:
    annot_help = st.button(help_btn, key=(TAB_SEARCH + "_help"), on_click=toggle_help)
    if not st.session_state.get("help_hidden", True):
        st.info("""**Help (Search):**   
        •  Search for more projects from the full BioProject database, to supplement the local dataset and strengthen your annotations.   
        •  Add the search results to the dataset so you can start annotating them in the **Annotate** tab.   
        •  If you don't wish to add everything, click on the specific projects you want to select.""")
        
    st.header("Search BioProject")
    
    api_terms = display_search_feature(TAB_SEARCH)
    if api_terms:
        api_df, api_pub_df, found = api_search(api_terms, project_df[[UID_COL, ACC_COL]].values)
        if api_df is not None and not api_df.empty:
            st.write(f"Results for '{api_terms}':")
            display_interactive_grid(TAB_SEARCH, api_df, search_columns)
            display_add_to_dataset_feature(TAB_SEARCH, connection, api_df, api_pub_df)
        elif found:
            st.write(f"No results for '{api_terms}' that are not already in the dataset. Try looking for something else.")
        else:
            st.write(f"No results for '{api_terms}'. Check for typos or try looking for something else.")


# Predict tab  
with predict:
    annot_help = st.button(help_btn, key=(TAB_PREDICT + "_help"), on_click=toggle_help)
    if not st.session_state.get("help_hidden", True):
        st.info(f"""**Help (Predict):**   
        •  Run the prediction algorithm to get predictions for all unannotated projects (based on your existing annotations), each assigned a score for the confidence in that prediction.    
        •  You can confirm or correct predictions by annotating manually.   
        •  To best improve the future performance of the algorithm, annotate the projects suggested below.""")
        
    st.header("Predict annotations")
    
    message = st.empty()
    message.write("Click **Start** to get predictions for all unannotated projects.")
    start_button = st.button("Start", key="start_button")

    if start_button:
        X_labelled, X_unlabelled, y_labelled, labels, error = process_dataset(project_df, pub_df, text_columns)
        
        if X_labelled:
            train_size = len(X_labelled)
            test_size = len(X_unlabelled)
            dataset_hash = hashlib.shake_256(pd.util.hash_pandas_object(project_df[[ACC_COL, ANNOT_COL]].sort_values(ACC_COL, axis=0), index=False).values).hexdigest(8)
            
            y_predicted, y_scores, f1_micro_ci, f1_macro_ci = active_learning(X_labelled, X_unlabelled, y_labelled)

            st.session_state.f1_micro_ci = f1_micro_ci
            st.session_state.f1_macro_ci = f1_macro_ci
            
            # Columns irrelevant to method cacheing dropped
            df = project_df.drop([PREDICT_COL, LEARN_COL], axis=1)
            process_predictions(y_predicted, y_scores, labels, df, connection)
            
            metric_row = np.array([date.today().strftime("%d/%m/%Y"), dataset_hash, train_size, test_size, np.mean(f1_micro_ci), np.mean(f1_macro_ci)])
            if metric_df.empty or not (metric_df == metric_row).all(1).any():
                insert_sheet(connection, metric_row, metric_columns, GSHEETS_URL_METRICS)
                metric_df.loc[len(metric_df)] = metric_row

            st.session_state.new_predictions = True
        else:
            message.write(error)
    
    # Prediction could be empty so relying on existence of score
    predict_df = project_df[project_df[SCORE_COL].replace('', None).notnull()].reset_index(drop=True)
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
        display_annotation_feature(TAB_PREDICT, connection, predict_df)
    
    learn_section_name = "Learn"
    learn_df = project_df[project_df[LEARN_COL].replace('', None).notnull()]    
    if not learn_df.empty:
        st.header("Improve prediction algorithm")
        st.write("To optimally improve performance, consider annotating the following projects:")
        # Sort by annotation importance to active learning
        learn_df = learn_df.sort_values(LEARN_COL, axis=0, ignore_index=True, key=lambda col: int_column(col))
        display_interactive_grid(learn_section_name, learn_df, annot_columns)
        display_annotation_feature(learn_section_name, connection, learn_df)

        
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