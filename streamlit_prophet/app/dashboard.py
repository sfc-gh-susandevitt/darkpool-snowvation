from typing import Any, Dict, List

import streamlit as st
import snowflake.connector

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
with st.expander("What is darkpool?", expanded=True):
    st.write(readme["app"]["app_intro"])
    st.write("")
st.write("")
st.sidebar.image(load_image("darkpool.png"), use_column_width=True)
#display_links(readme["links"]["repo"],readme["links"]["repo"])

#Snowflake Connection
#!/usr/bin/env python3

# Initialize connection.
# Uses st.cache to only run once.

# Headers   

st.header("Your Selections:")
st.write("")








#Sidebar

st.sidebar.title("Configure your analysis")
st.sidebar.caption("Snowflake Database = SNOWCAT2")

#add these back in
def init_connection():
    return snowflake.connector.connect(**st.secrets["snowflake"])

conn = init_connection()

#Select Table

def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)

        # Return a Pandas DataFrame containing all of the results.
        df = cur.fetch_pandas_all()
        #st.dataframe(df)
        #labels = df[‘’].unique()
        option = st.sidebar.selectbox('Select your dataset', df)
        st.write('You have selected dataset ',option)
 
run_query("select concat(TABLE_CATALOG,'.',TABLE_SCHEMA,'.',TABLE_NAME) from DEMAND.INFORMATION_SCHEMA.TABLES where TABLE_SCHEMA not in ('INFORMATION_SCHEMA');") 



#Select Dependent Variable

def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)

        # Return a Pandas DataFrame containing all of the results.
        df = cur.fetch_pandas_all()
        #st.dataframe(df)
        #labels = df[‘’].unique()
        option2 = st.sidebar.selectbox('Select your dependent variable', df)
        st.write('You have selected dependent variable ',option2)

text1 = "select COLUMN_NAME from DEMAND.INFORMATION_SCHEMA.COLUMNS where concat(TABLE_CATALOG,'.',TABLE_SCHEMA,'.',TABLE_NAME) = '"
text2 = "DEMAND.DATA.CUSTOMERS"
#text2 = st.write(option2)
text3 = "' order by 1 asc;"        
query_text = text1+text2+text3
run_query(query_text)  

#Analyze boost
st.subheader("Analyze potential boost?")
st.write("")

with st.expander("Click here for boost analysis",expanded=False):
    st.write("Here is your boost analysis:")
## Add chart







# Boost
st.header("Auto-boost your models?")
boost=st.button('Yes')



# Launch analysis
#with st.expander("Boost", expanded=True):
#    st.write("Choose the data sets for your analysis:")
#analysis = st.radio ("",('Off','On'))
#if analysis == 'Off': 
#    st.write('You selected ML analysis on your own data set only.  Here are your results.')
    
#Add metics
#if analysis == 'On': 
#    st.write('You selected to boost your ML accuracy with data from the dark pool. Here are your results')
        
# Launch analysis
#if st.sidebar.button ('Off'):
#   st.write('You selected ML analysis on your own data set only.')
    
    



        




