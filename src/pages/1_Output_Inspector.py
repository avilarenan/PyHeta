import streamlit as st
import time
import numpy as np
from st_files_connection import FilesConnection
import streamlit.components.v1 as components
import pandas as pd
from functools import reduce
from streamlit_tree_select import tree_select
from st_ant_tree import st_ant_tree
import os
import datetime
from sqlalchemy import create_engine
from static_info_utils import POSTGRES_CONN_STRING_TRAIN
import plotly.express as px


st.markdown("# Inspector")

engine = create_engine(POSTGRES_CONN_STRING_TRAIN)
conn = engine.connect().execution_options(stream_results=True)

df_list = []

for chunk_dataframe in pd.read_sql(""" SELECT * FROM public."Result_metrics" """, conn, chunksize=50000):
    df_list += [chunk_dataframe]

df_result_metrics = pd.concat(df_list)

conn.close()