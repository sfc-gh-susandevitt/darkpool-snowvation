from typing import Any, Dict, List

import streamlit as st
from streamlit_prophet.lib.dataprep.clean import clean_df
from streamlit_prophet.lib.dataprep.format import (
    add_cap_and_floor_cols,
    check_dataset_size,
    filter_and_aggregate_df,
    format_date_and_target,
    format_datetime,
    print_empty_cols,
    print_removed_cols,
    remove_empty_cols,
    resample_df,
)
from streamlit_prophet.lib.dataprep.split import get_train_set, get_train_val_sets
from streamlit_prophet.lib.exposition.export import display_links, display_save_experiment_button
from streamlit_prophet.lib.exposition.visualize import (
    plot_components,
    plot_future,
    plot_overview,
    plot_performance,
)
from streamlit_prophet.lib.inputs.dataprep import input_cleaning, input_dimensions, input_resampling
from streamlit_prophet.lib.inputs.dataset import (
    input_columns,
    input_dataset,
    input_future_regressors,
)
from streamlit_prophet.lib.inputs.dates import (
    input_cv,
    input_forecast_dates,
    input_train_dates,
    input_val_dates,
)
from streamlit_prophet.lib.inputs.eval import input_metrics, input_scope_eval
from streamlit_prophet.lib.inputs.params import (
    input_holidays_params,
    input_other_params,
    input_prior_scale_params,
    input_regressors,
    input_seasonality_params,
)
from streamlit_prophet.lib.models.prophet import forecast_workflow
from streamlit_prophet.lib.utils.load import load_config, load_image

# Page config
st.set_page_config(page_title="darkpool", layout="wide")

# Load config
config, instructions, readme = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)

# Initialization
dates: Dict[Any, Any] = dict()
report: List[Dict[str, Any]] = []

# Info
with st.beta_expander("What is darkpool?", expanded=True):
    st.write(readme["app"]["app_intro"])
    st.write("")
st.write("")
st.sidebar.image(load_image("darkpool.png"), use_column_width=True)
#display_links(readme["links"]["repo"],readme["links"]["repo"])



st.sidebar.title("Configure your analysis")

# Select Dataset
with st.sidebar.beta_expander("Data", expanded=True):
    dataset = st.selectbox('Select your dataset for analysis',('Credit Card Fraud','Churn'))

# Column names - change to target variable
with st.sidebar.beta_expander("Columns", expanded=True):
    if dataset == 'Credit Card Fraud':   
        column = st.selectbox('Select your target outcome variable',('ISFRAUD','ISFLAGGEDFRAUD'))
    if dataset == 'Churn':   
        st.write("No dataset is available")

# Launch analysis
with st.sidebar.beta_expander("Boost", expanded=True):
#    st.write("Choose the data sets for your analysis:")
    analysis = st.radio ("Choose the data sets for your analysis:",('None','Own Set','BOOST'))
    if analysis == 'Own Set': 
        st.write('You selected ML analysis on your own data set only.')
    if analysis == 'BOOST': 
        st.write('You selected to boost your ML accuracy with data from the dark pool.')
        

# Visualizations        
 
st.header("1. Overview (visualization of data)")
st.write("")
st.header("2. Evaluation on Dataset")
st.subheader("Performance Metrics")
with st.beta_expander("More info on evaluation metrics",expanded=True):
    st.write(readme["plots"][ "metrics"])
st.write("")
st.header("3. Impact of components and regressors")
st.write("")

