from ast import literal_eval
from collections import Counter

import streamlit as st
from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler
from src.nsga2_svm_fs.nsga2_svmfs import NSGA2_SVMFS
from src.utils.visualization import plot_pareto_front

# App title
st.title("🌱 NSGA-II for Feature Selection with SVM")

tab1, tab2 = st.tabs(["⚙️ Simulation", "📊 Results"])

with tab1:
    st.header("Simulation")
    # File upload
    uploaded_file = st.file_uploader("📂 Upload your CSV or TXT file", type=["csv", "txt"])

    if uploaded_file:

        # === CSV Read Options ===
        st.subheader("⚙️ CSV Read Options")
        has_header = st.checkbox("Does the file have a header?", value=True)
        has_index = st.checkbox("Does the file have an index column?", value=False)
        has_costs = st.checkbox("Does the file include a row with feature costs?", value=False)
        separator = st.selectbox("Select separator", [",", ";", "\t"], index=1)
        normalize = st.checkbox("📏 Normalize input features?", value=False)

        # CSV parsing configuration
        header_option = 0 if has_header else None
        index_col_option = 0 if has_index else None

        try:
            # === Read and preprocess the CSV ===
            data = read_csv(uploaded_file, sep=separator, header=header_option)

            # Remove index column if indicated
            if has_index:
                data.drop(columns=[data.columns[0]], inplace=True)

            # If no header is provided, assign default column names
            if not has_header:
                num_cols = data.shape[1]
                column_names = ['y'] + [f'x{i}' for i in range(1, num_cols)]
                data.columns = column_names
            else:
                # Reorder columns to ensure the output (label) is first
                data = data[[data.columns[0]] + list(data.columns[1:])]

            # Set input/output columns
            output_col = data.columns[0]
            input_cols = data.columns[1:].tolist()

            # Extract cost row if applicable
            if has_costs:
                costs = data.iloc[0].values[1:].tolist()
                data = data.drop(index=data.index[0])
            else:
                costs = [0.0] * len(input_cols)

            # Normalize input features
            if normalize:
                scaler = MinMaxScaler()
                scaled_features = scaler.fit_transform(data[input_cols])
                data_scaled = DataFrame(scaled_features, columns=input_cols)
                data_scaled[output_col] = data[output_col].values
            else:
                data_scaled = data.copy()

            # Show sample of processed data
            st.success("✅ File loaded successfully.")
            st.dataframe(data_scaled.head())

        except Exception as e:
            st.error(f"❌ Error while loading the CSV: {e}")
            st.stop()

        # === Algorithm Configuration ===
        method = st.selectbox("⚙️ Dominance method", ["DIST-EPS", "DIST-EPS-COST", "MC", "MC-COST"])
        population_size = st.number_input("👥 Population size", min_value=5, value=10, max_value=100)
        num_features = st.number_input("🔢 Number of features to select", min_value=1, max_value=len(input_cols),
                                       value=5)
        iter_type = st.radio("⏱ Training mode", ["iter", "sec"])
        iter_val = st.number_input(f"Number of {iter_type}", min_value=1, value=5)

        # === Run the algorithm ===
        if st.button("▶️ Run simulation"):
            with st.spinner("⏳ Running NSGA-II..."):
                log_area = st.empty()
                logs = []


                def log(message: str):
                    """Log messages to the UI in real-time."""
                    logs.append(message)
                    log_area.text_area("📜 Execution log", value="\n".join(logs), height=200)


                # Initialize and run the algorithm
                algorithm = NSGA2_SVMFS(
                    method=method,
                    data=data_scaled,
                    costs=costs,
                    population_size=population_size,
                    inputs=input_cols,
                    output=output_col,
                    num_selected_features=num_features,
                    logger=log  # <- Custom logger for Streamlit
                )

                algorithm.run(train=iter_type, num_iter=iter_val)


with tab2:
    st.header("Results and visualization")

    # === Results ===
    results = algorithm.population.solutions_df
    pareto_front = results[results.FRONT == 0]

    st.success("✅ Simulation completed.")
    st.dataframe(pareto_front)

    # === Visualization ===
    st.subheader("🌐 Pareto Front Visualization")
    try:
        fig = plot_pareto_front(pareto_front, method)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"❌ Could not generate the plot: {e}")

    st.subheader("📊 Most Common Features and Vectors in Pareto Front")

    # Parse string representations to real lists if needed
    def parse_column_as_list(series):
        if isinstance(series.iloc[0], str):
            return series.apply(literal_eval)
        return series

    # Convert FEATURES and VECTORS columns to list objects
    features_list_series = parse_column_as_list(pareto_front['FEATURES'])
    vectors_list_series = parse_column_as_list(pareto_front['VECTORS'])

    # Count feature frequency (flatten all features into one list)
    feature_counts = Counter(f for features in features_list_series for f in features)
    most_common_features = feature_counts.most_common()

    # Count full vector combinations (as tuples to make them hashable)
    vector_counts = Counter(tuple(vec) for vec in vectors_list_series)
    most_common_vectors = vector_counts.most_common()

    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Mostrar features a la izquierda
    with col1:
        st.markdown("### 🧬 Most Frequent Features")
        for feature, count in most_common_features[:10]:
            st.markdown(f"- **{feature}**: {count} times")

    # Mostrar vectores a la derecha
    with col2:
        st.markdown("### 🔁 Most Frequent Vectors")
        for vector, count in most_common_vectors[:10]:
            st.markdown(f"- `{vector}`: {count} times")

    metrics_cols = ['SOL', 'ACCURACY', 'F1', 'KAPPA', 'AUC', 'COST']

    st.subheader("📊 Metrics of solutions")

    # Mostrar sólo las columnas deseadas (si existen en el dataframe)
    metrics_cols = [col for col in metrics_cols if col in pareto_front.columns]
    st.dataframe(pareto_front[metrics_cols])