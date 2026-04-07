from pathlib import Path

DATA_DIR = Path("data/mit-bih-arrhythmia-database-1.0.0")

# Start with one record for reliability
RECORDS = ["100"]

LEFT_WINDOW = 100
RIGHT_WINDOW = 100

TEST_SIZE = 0.2
RANDOM_STATE = 42

NUM_CLIENTS = 3
NUM_ROUNDS = 3
LOCAL_EPOCHS = 2
LEARNING_RATE = 1e-3

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
