import streamlit as st
import pandas as pd
import pika
from sqlalchemy import create_engine
import json
from static_info_utils import POSTGRES_CONN_STRING_DATA, JSON_CREDENTIALS, AMQP_HOST
import itertools
import plotly.express as px

st.set_page_config(
    page_title="Processed Data",
    page_icon="📈",
    layout="wide"
)

credentials = pika.PlainCredentials(JSON_CREDENTIALS["AMQP_USER"], JSON_CREDENTIALS["AMQP_PASSWORD"])

engine = create_engine(POSTGRES_CONN_STRING_DATA)
conn = engine.connect().execution_options(stream_results=True)

df_list = []

for chunk_dataframe in pd.read_sql(""" SELECT * FROM Public."Datasets" """, conn, chunksize=50000):
    df_list += [chunk_dataframe]

df_datasets = pd.concat(df_list)
conn.close()

st.markdown("# Hyperparameter Grouped DataSets")
select_all = st.toggle("Select all")

df_datasets_grouped = df_datasets.groupby(
    ["dataset", "farm_fuzzyfy", "farm_binary_shaping", "farm_ffalign", "skip_ceemdan", "skip_farm_shaping"],
    dropna=False
).agg({
    "dataset_id" : lambda x: list(set(list(x))),
    "unique_id" : lambda x: list(set(list(x)))
})
df_datasets_grouped = df_datasets_grouped.reset_index()
df_datasets_grouped.insert(loc=0, column="Selected", value=False)
df_datasets_grouped["Selected"] = True if select_all else False

unique_dataset_types = list(df_datasets_grouped["dataset"].unique())
selected_dataset_types = st.multiselect(
    label="Select dataset types",
    options=unique_dataset_types,
    default=unique_dataset_types 
)
df_datasets_grouped = df_datasets_grouped[df_datasets_grouped["dataset"].isin(selected_dataset_types)]
edited_df = st.data_editor(df_datasets_grouped)

st.text(f"""Number of selected groups: {len(edited_df[edited_df["Selected"] == True])}""")
selected_datasets = list(itertools.chain.from_iterable(edited_df[edited_df["Selected"] == True]["dataset_id"].tolist()))
st.text(f"""Number of Datasets in selected groups: {len(selected_datasets)}""")
st.text(f"""Number of unique Datasets in selected groups: {len(list(set(selected_datasets)))}""")

st.markdown("## Selected datasets")

with st.expander("Full Selected Data Sets"):
    filtered_df = df_datasets[df_datasets["dataset_id"].isin(selected_datasets)]
    st.dataframe(filtered_df)
    st.text(f"Length: {len(filtered_df)}")

st.markdown("### Deduplicated keeping the most recent")
params_subset_columns = ["farm_fuzzyfy", "farm_binary_shaping", "farm_ffalign", "skip_ceemdan", "skip_farm_shaping", "dataset", "features", "target_feature"]
dedup_df = filtered_df.sort_values("created_at")
dedup_df = dedup_df.drop_duplicates(subset=params_subset_columns, keep="last")

st.divider()

tab1, tab2 = st.tabs(["Launch training", "Processed data inspection"])

with tab1:

    dedup_df.insert(loc=0, column="Selected", value=True)
    select_all = st.toggle("Select all experiments to run")
    dedup_df["Selected"] = select_all
    edited_dedupped_df = st.data_editor(dedup_df)
    st.text(f"Length: {len(dedup_df)}")
    st.text(f"""Selected: {len(edited_dedupped_df[edited_dedupped_df["Selected"] == True])}""")

    only_shaped_features_selection = st.checkbox("ONLY_SHAPED_FEATURES", False)
    only_target_feature_for_selected_imfs_selection = st.checkbox("ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS", True)
    only_target_selection = st.checkbox("ONLY_TARGET", False)
    only_original_features = st.checkbox("ONLY_ORIGINAL_FEATURES", False)

    repetitions = st.number_input("Repetitions", 1)

    if st.button("Launch multiple trainings", type="primary"):
        connection = pika.BlockingConnection(pika.ConnectionParameters(AMQP_HOST, credentials=credentials))
        channel = connection.channel()
        selected_datasets_list = edited_dedupped_df[edited_dedupped_df["Selected"] == True]["dataset_id"].tolist()

        for current_dataset_id in selected_datasets_list:

            filtered_df = df_datasets[df_datasets["dataset_id"] == current_dataset_id]

            target_feature_name = list(filtered_df["target_feature"].unique())[0]
            list_of_features = list(filtered_df["features"].unique())[0][1:-1].split(",")

            ablation_id = list(filtered_df["ablation_id"].unique())[0]

            train_instruction = {
                'ablation_id': ablation_id,
                'dataset_id': current_dataset_id,
                'DATASET' : list(filtered_df["dataset"].unique())[0],
                'target_feature' : target_feature_name,
                'SKIP_CEEMDAN': list(filtered_df["skip_ceemdan"].unique())[0],
                'SKIP_FARM': False,
                'SKIP_FARM_SHAPING': list(filtered_df["skip_farm_shaping"].unique())[0],
                'ONLY_SHAPED_FEATURES': only_shaped_features_selection,
                'ONLY_TARGET_FEATURE_FOR_SELECTED_IMFS': only_target_feature_for_selected_imfs_selection,
                'ONLY_ORIGINAL_FEATURES': only_original_features,
                'ONLY_TARGET': only_target_selection,
                'farm_ffalign' : list(filtered_df["farm_ffalign"].unique())[0],
                'farm_fuzzyfy' : list(filtered_df["farm_fuzzyfy"].unique())[0],
                'farm_binary_shaping': list(filtered_df["farm_binary_shaping"].unique())[0],
                'latter_imfs' : ["IMF3", "IMF4", "IMF5", "IMF6", "IMF7", "IMF8", "IMF9", "Residue"],
                'initial_imfs' : ["IMF0", "IMF1", "IMF2"],
                'FILTER_FEATURES' : list_of_features,
                'ablation_features' : list_of_features,
                'sequence_length' : 8,
                'windows': {
                    "IMF0" : 2,
                    "IMF1" : 3,
                    "IMF2" : 4,
                    "IMF3" : 4,
                    "IMF4" : 5,
                    "Residue" : 6,
                },
                'learning_rate' : 0.001,
                'learning_rate_by_imf' : {
                    "IMF0": 0.0010,
                    "IMF1": 0.0010,
                    "IMF2": 0.0010,
                    "IMF3": 0.0010,
                    "IMF4": 0.0010,
                    "Residue": 0.0001
                },
                'batch_size' : 64,
                'n_epochs' : 500,
                'n_epochs_by_imf' : {
                    "IMF0": 1000,
                    "IMF1": 1000,
                    "IMF2": 1000,
                    "IMF3": 500,
                    "IMF4": 500,
                    "Residue": 10000
                },
                'n_epochs_stop' : 5,
                'n_epochs_stop_by_imf' : {
                    "IMF0": 5,
                    "IMF1": 10,
                    "IMF2": 10,
                    "IMF3": 10,
                    "IMF4": 10,
                    "Residue": 50
                },
                'model_by_imf' : {
                    "IMF0": "LSTM",
                    "IMF1": "LSTM",
                    "IMF2": "LSTM",
                    "IMF3": "LSTM",
                    "IMF4": "LSTM",
                    "Residue": "MLP"
                },
                'default_model' : "LSTM",
                'min_delta_stop': 0.0000,
                'background_data_size' : 900,
                'test_sample_size' : 100,
                'repetitions' : int(repetitions),
                'save_files_to_s3' : True,
                's3_path' : f"/{ablation_id}/",
                's3_bucket' : "forecastingexperimentstrain",
            }

            channel.basic_publish(
                exchange='',
                routing_key='train_instructions',
                body=json.dumps(train_instruction, default=str)
            )

        connection.close()

with tab2:
    st.markdown("Choose data set to inspect")

    dedup_df = dedup_df.drop("Selected", axis=1)
    event = st.dataframe(
        dedup_df,
        on_select="rerun",
        selection_mode=["single-row"],
    )
    selected_row = dedup_df.iloc[event.selection.rows]

    st.text(f"Length: {len(dedup_df)}")

    st.markdown("Selected row")
    st.dataframe(selected_row)

    inspected_dataset_unique_ids = list(selected_row["unique_id"])
    if len(inspected_dataset_unique_ids) == 1:
        inspected_dataset_unique_id = inspected_dataset_unique_ids[0]
        st.markdown(f"Inspected dataset id: {inspected_dataset_unique_id}")

        engine = create_engine(POSTGRES_CONN_STRING_DATA)
        conn = engine.connect().execution_options(stream_results=True)

        df_list = []
        for chunk_dataframe in pd.read_sql(f""" SELECT * FROM public."{inspected_dataset_unique_id}" """, conn, chunksize=50000):
                df_list += [chunk_dataframe]

        df_dataset_content = pd.concat(df_list)

        conn.close()
        
        
        with st.expander("Inspected dataset content"):
            st.markdown("Data Set Content")
            st.dataframe(df_dataset_content)

        col1, col2 = st.columns(2)

        with col1:
            x_axis_selection = st.selectbox(label="Select X axis", options=list(df_dataset_content.columns))
        with col2:
            y_axis_selection = st.multiselect(label="Select Y axis", options=list(df_dataset_content.columns))

        fig = px.line(df_dataset_content, x=x_axis_selection, y=y_axis_selection)

        st.plotly_chart(fig, key="graph_final_processed data_inspection")

    else:
        print("No selected dataset, or multiple selected datasets. Expects only one dataset to be selected.")


