import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (
    DATA_DIR,
    RECORDS,
    LEFT_WINDOW,
    RIGHT_WINDOW,
    TEST_SIZE,
    RANDOM_STATE,
    NUM_CLIENTS,
)


def load_record(record_id: str):
    path = str(DATA_DIR / record_id)
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")
    signal = record.p_signal[:, 0]
    return signal, annotation


def extract_beats_from_record(record_id: str):
    signal, annotation = load_record(record_id)
    X, y = [], []

    for idx, sym in zip(annotation.sample, annotation.symbol):
        start = idx - LEFT_WINDOW
        end = idx + RIGHT_WINDOW

        if start >= 0 and end < len(signal):
            beat = signal[start:end]
            if len(beat) == LEFT_WINDOW + RIGHT_WINDOW:
                X.append(beat)
                y.append(sym)

    return np.array(X, dtype=np.float32), np.array(y)


def load_dataset():
    all_X, all_y = [], []

    for record_id in RECORDS:
        X, y = extract_beats_from_record(record_id)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        raise FileNotFoundError(
            f"No usable MIT-BIH records found in {DATA_DIR}. Check dataset path."
        )

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y, y_encoded, label_encoder


def get_train_test_data():
    X, y_raw, y_encoded, label_encoder = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    return X_train, X_test, y_train, y_test, y_raw, label_encoder


def split_into_clients(X, y, num_clients=NUM_CLIENTS):
    X_splits = np.array_split(X, num_clients)
    y_splits = np.array_split(y, num_clients)
    return list(zip(X_splits, y_splits))
