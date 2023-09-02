import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from features.gsheets import get_gsheets_columns
from features.search import get_base_urls

BASE_URL_PROJ, BASE_URL_PUB = get_base_urls()

project_columns, pub_columns, metric_columns = get_gsheets_columns()

UID_COL, ACC_COL, TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL, ANNOT_COL, PREDICT_COL, SCORE_COL, LEARN_COL  = project_columns

detail_columns = [ACC_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL]
export_columns = [UID_COL, ACC_COL, TITLE_COL, NAME_COL, DESCR_COL, TYPE_COL, SCOPE_COL, ORG_COL, PUB_COL, ANNOT_COL, PREDICT_COL]

PLACEHOLDER = "-"
RESULTS_PER_PAGE = 10

reload_btn = "↻"
export_btn = "Export to CSV"
show_btn = "View details"
hide_btn = "Hide details"
next_btn = "Next"
prev_btn = "Previous"

markdown_translation = str.maketrans({char : '\\' + char for char in r'\`*_{}[]()#+-.!:><&'})

primary_colour = "#81b1cc"
aggrid_css = {
    "#gridToolBar": {"display": "none;"},
    ".ag-theme-alpine, .ag-theme-alpine-dark": {"--ag-font-size": "12px;"},
    ".ag-cell": {"padding": "0px 12px;"},
}


def get_grid_buttons():
    return reload_btn, export_btn, show_btn, hide_btn, next_btn, prev_btn
    
    
def get_primary_colour():
    return primary_colour


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
    # Indicates that a project was selected on the grid, trigerring an experimental rerun, with session state variables updated in order to display that selection
    rerun = st.session_state.get(tab + "_rerun", False)
    
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
    
    # Get selection from grid
    selected_row = grid['selected_rows']
    selected_df = pd.DataFrame(selected_row)
    previous_page = st.session_state.get(tab + "_starting_page", 0)
    project_details_hidden = st.session_state.get(tab + "_project_details_hidden", True)
    
    # Export button
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Export to CSV", df[export_columns].rename(columns={ANNOT_COL:"Manual_Annotation", PREDICT_COL: "Predicted_Annotation"}).to_csv(index=False).encode('utf-8'), "BioProjct_Annotation.csv", "text/csv", key=(tab + "_export"))
    with col2:
        if project_details_hidden:
            st.button(show_btn, key=(tab + "_show"), on_click=show_details, args=(tab,))
        else:
            st.button(hide_btn, key=(tab + "_hide"), on_click=hide_details, args=(tab,))
        
    # If this is a rerun, display new selection
    if rerun:
        if not project_details_hidden:
            display_project_details(df.iloc[selected_row_index])
            if nav_buttons:
                display_navigation_buttons(tab, len(df))
            
        st.session_state[tab + "_starting_page"] = starting_page
        st.session_state[tab + "_rerun"] = False

    # Else if a selection was just made, update the session state with the relevant details and rerun to display it
    elif not selected_df.empty:
        selected_mask = df[ACC_COL].isin(selected_df[ACC_COL])
        selected_data = df.loc[selected_mask]
        
        selected_row_index = selected_data.index.tolist()[0]
        st.session_state[tab + "_selected_row_index"] = selected_row_index
        
        selected_id = df.iloc[selected_row_index][ACC_COL]
        selected_projects = st.session_state.get(tab + "_selected_projects", [])
        if selected_id not in selected_projects:
            st.session_state[tab + "_selected_projects"] = selected_projects + [selected_id]
            
        st.session_state[tab + "_rerun"] = True
        
        # Rerun to have selection displayed
        st.experimental_rerun()   
    
    # Else display the details of the latest project to be selected
    elif not project_details_hidden:
        display_project_details(df.iloc[selected_row_index])
        if nav_buttons:
            display_navigation_buttons(tab, len(df))
            
    st.write("")