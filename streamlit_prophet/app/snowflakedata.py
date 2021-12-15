#!/usr/bin/env python3

import streamlit as st
import snowflake.connector

# Initialize connection.
# Uses st.cache to only run once.
#@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    return snowflake.connector.connect(**st.secrets["snowflake"])

conn = init_connection()

# Perform query.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

rows = run_query("select case when start_station_id = 3118 then 'Demand' else 'Supply'end as DATASET_PROVIDER_TYPE, start_station_id as DATASET_ID, count (*) as records from trips group by start_station_id order by DATASET_PROVIDER_TYPE, start_station_id;")

# Print results.
for row in rows:
    st.write(f"{row[0]} with dataset id {row[1]} has {row[2]} rows")
