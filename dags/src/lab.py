import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os


def _project_root():
    # /opt/airflow/dags when mounted via compose
    return os.path.dirname(os.path.dirname(__file__))


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        bytes: Serialized data.
    """
    df_path = os.path.join(_project_root(), "data", "file.csv")
    df = pd.read_csv(df_path)
    serialized_data = pickle.dumps(df)
    return serialized_data


def data_preprocessing(data):
    """
    Deserializes data, performs data preprocessing, and returns serialized, SCALED data.
    Args:
        data (bytes): Serialized data to be deserialized and processed.
    Returns:
        bytes: Serialized scaled numpy array for clustering.
    """
    df = pickle.loads(data)
    df = df.dropna()
    cols = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
    clustering_data = df[cols].copy()

    # Scale here for model-building; we will scale test with train-fit in load step.
    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    return pickle.dumps(clustering_data_minmax)


def build_save_model(data, filename):
    """
    Builds a KMeans model, chooses K via elbow, saves it, and returns SSE list.
    Args:
        data (bytes): Serialized SCALED data for clustering.
        filename (str): Output filename (e.g., 'model.sav').
    Returns:
        list: SSE values for k=1..49
    """
    X = pickle.loads(data)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    last_model = None
    for k in range(1, 50):
        km = KMeans(n_clusters=k, **kmeans_kwargs)
        km.fit(X)
        sse.append(km.inertia_)
        last_model = km

    # pick elbow; fall back to 3 if Knee can't find one
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    k_opt = kl.elbow or 3

    best_model = KMeans(n_clusters=k_opt, **kmeans_kwargs).fit(X)

    output_dir = os.path.join(_project_root(), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(best_model, f)

    return sse


def load_model_elbow(filename, sse):
    """
    Loads the saved KMeans model and predicts cluster for the single test row.
    Also computes elbow from provided SSE for logging/debug.
    Args:
        filename (str): Saved model file name.
        sse (list): SSE values returned from build_save_model.
    Returns:
        int: Predicted cluster label for the test row.
    """
    model_path = os.path.join(_project_root(), "model", filename)
    loaded_model = pickle.load(open(model_path, "rb"))

    # Fit scaler on TRAIN columns from file.csv (same columns used earlier)
    train_path = os.path.join(_project_root(), "data", "file.csv")
    test_path = os.path.join(_project_root(), "data", "test.csv")
    train_df = pd.read_csv(train_path)[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]].dropna()
    test_df = pd.read_csv(test_path)[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    scaler = MinMaxScaler().fit(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters (elbow): {kl.elbow or 3}")

    preds = loaded_model.predict(test_scaled)
    return int(preds[0])
