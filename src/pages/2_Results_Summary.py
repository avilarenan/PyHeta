import boto3
import pandas as pd
from io import StringIO
import json
import streamlit as st
from collections.abc import MutableMapping
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine
import json
import plotly.express as px
import traceback
from static_info_utils import POSTGRES_CONN_STRING_TRAIN

client = boto3.client('s3')

st.set_page_config(
    page_title="Results Summary",
    page_icon="📈",
    layout="wide"
)

st.markdown("# Results summary")



engine = create_engine(POSTGRES_CONN_STRING_TRAIN)
conn = engine.connect().execution_options(stream_results=True)

df_list = []

for chunk_dataframe in pd.read_sql(""" SELECT * FROM public."Result_metrics" """, conn, chunksize=50000):
    df_list += [chunk_dataframe]

df_result_metrics = pd.concat(df_list)



column_summary = st.toggle("Column summary")

definition_columns = ["dataset", "features", "target_feature", "skip_ceemdan", "skip_farm_shaping", "farm_ffalign", "farm_fuzzyfy", "farm_binary_shaping"]
metrics_columns = ["final_rmse_test", "final_mape_test", "final_mae_test", "final_rmse_train", "final_mape_train", "final_mae_train", "run_train_unique_id", "dataset_id"]
summarized_columns = definition_columns + metrics_columns

dataset_filter = st.multiselect(
    label="Datasets",
    options=list(df_result_metrics["dataset"].unique()),
    default=list(df_result_metrics["dataset"].unique()),
)

filter_skip_farm = None
filter_ceemdan = None

enable_skip_farm_filter = st.toggle("Enable Column Filters")
if enable_skip_farm_filter:
    filter_row_1_colum_1, filter_row_1_column_2 = st.columns(2) 
    with filter_row_1_colum_1:
        filter_skip_farm = st.radio("Skip Farm", ["True", "False", "Any"])
    with filter_row_1_column_2:
        filter_ceemdan = st.radio("Skip Ceemdan", ["True", "False", "Any"])

df = df_result_metrics.sort_values(by="created_at")

df = df_result_metrics[summarized_columns] if column_summary else df_result_metrics

df = df[df["dataset"].isin(dataset_filter)]

if enable_skip_farm_filter and filter_skip_farm in ["True", "False"]:
    df = df[df["skip_farm_shaping"] == filter_skip_farm]

if enable_skip_farm_filter and filter_ceemdan in ["True", "False"]:
    df = df[df["skip_ceemdan"] == filter_ceemdan]

dedup_df = df.drop_duplicates(subset=definition_columns, keep="last")

event = st.dataframe(
    dedup_df,
    on_select="rerun",
    selection_mode=["single-row"],
)

st.text(f"Total length: {len(dedup_df)}")

with st.expander("Debug info"):

    st.markdown("### Grouped by features")
    grouped_counted_df_features = dedup_df.groupby("features", dropna=False).count()
    st.dataframe(grouped_counted_df_features)
    st.text(f"Length = {len(grouped_counted_df_features)}")

    st.markdown("### Grouped by parameters")
    grouped_counted_df_by_parameters = dedup_df.groupby(["dataset", "skip_ceemdan", "skip_farm_shaping", "farm_ffalign", "farm_fuzzyfy", "farm_binary_shaping"], dropna=False).count()
    st.dataframe(grouped_counted_df_by_parameters)
    st.text(f"Length = {len(grouped_counted_df_by_parameters)}")

selected_row = dedup_df.iloc[event.selection.rows]
st.markdown("## Selected row inspection")
st.write(selected_row)

if not selected_row.empty:
    run_train_unique_ids = list(selected_row["run_train_unique_id"])
    if len(run_train_unique_ids) == 0:
        st.write("No row selected")
        run_train_unique_id = None
    elif len(run_train_unique_ids) == 1:
        run_train_unique_id = run_train_unique_ids[0]
    else:
        raise Exception(f"Unexpected number of run_train_unique_ids, len = {len(run_train_unique_ids)}")

    MAX_LEN_PLOT = st.number_input("MaxLenPlot", min_value=1000, max_value=10000, step=1000, value=5000)

    view_step = st.selectbox(
        label="View train step",
        options=[
            "step_6_component_error_metrics",
            "step_7_prediction",
            "step_9_recomposition"
        ]
    )

    df_list = []
    if run_train_unique_id is not None:
        # Index on run_train_unique_id necessary for tables
        for chunk_dataframe in pd.read_sql(f""" SELECT * FROM public."{view_step}" WHERE run_train_unique_id = '{run_train_unique_id}' """, conn, chunksize=50000):
            df_list += [chunk_dataframe]

        df_step_content = pd.concat(df_list)
        # df_step_content = df_step_content.reset_index()

        with st.expander("Full inspected row dataframe"):
            st.text("Under estriction of max len plot")
            st.dataframe(
                df_step_content[:MAX_LEN_PLOT],
            )

        try:

            selected_dataset_id = list(df_step_content["dataset_id"].unique())[0]
            filtered_df = dedup_df[dedup_df["dataset_id"] == selected_dataset_id]

            use_filter = st.toggle("Use filter")

            if use_filter:
                col1, col2 = st.columns(2)

                with col1:
                    filter_by = st.selectbox(label="Filter by", options=list(df_step_content.columns))
                with col2:
                    value_matching = st.selectbox(label="equals to", options=list(df_step_content[filter_by].unique()))

                df_step_content = df_step_content[df_step_content[filter_by] == value_matching]


            col3, col4 = st.columns(2)

            with col3:
                x_axis_selection = st.selectbox(label="Select X axis", options=list(df_step_content.columns))
            with col4:
                y_axis_selection = st.multiselect(label="Select Y axis", options=list(df_step_content.columns))

            st.text("Selected row parameters")

            df = filtered_df[summarized_columns] if column_summary else filtered_df
            st.dataframe(filtered_df, use_container_width=True)

            len_df_plot = len(df_step_content)
            
            st.text(f"Plot Max length: {MAX_LEN_PLOT}")

            if len_df_plot > MAX_LEN_PLOT:
                print("Overflow case")

                resample_step = int(len_df_plot/MAX_LEN_PLOT)
                resample_step = resample_step if resample_step > 1 else 1
                df_step_content_resampled_debug_copy = df_step_content.reset_index()[::resample_step].copy()
                df_step_content_resampled_debug_copy = df_step_content_resampled_debug_copy.drop("index", axis=1)
                df_step_content_resampled_debug_copy = df_step_content_resampled_debug_copy.reset_index()
                fig_full_resampled = px.line(df_step_content_resampled_debug_copy, x=x_axis_selection, y=y_axis_selection)

                with st.expander("Full resampled view for reference"):
                    st.plotly_chart(fig_full_resampled, key="graph_resampled")
                    toggle_show_dataframe = st.toggle("show debug resampled dataframe")
                    if toggle_show_dataframe:
                        st.dataframe(df_step_content_resampled_debug_copy)

                start_point = st.slider(
                    label="Start point",
                    min_value=0,
                    max_value=len_df_plot-MAX_LEN_PLOT,
                    value=0
                )
                df_step_content = df_step_content.drop("index", axis=1)
                df_step_content = df_step_content.reset_index().iloc[start_point:start_point+MAX_LEN_PLOT]
            
            else:
                df_step_content = df_step_content.drop("index", axis=1)
                df_step_content = df_step_content.reset_index()
                
            fig = px.line(df_step_content, x=x_axis_selection, y=y_axis_selection)

            st.plotly_chart(fig, key="graph_final")
            st.dataframe(df_step_content)
        except Exception as e:
            traceback.print_exc()

    conn.close()