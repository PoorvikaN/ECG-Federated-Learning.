import flwr as fl

from .config import NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, LEARNING_RATE, RESULTS_DIR
from .data_utils import get_train_test_data, split_into_clients
from .model import ECGModel, get_parameters, set_parameters, train_model, evaluate_model

X_train, X_test, y_train, y_test, _, label_encoder = get_train_test_data()
CLIENT_DATA = split_into_clients(X_train, y_train, NUM_CLIENTS)
NUM_CLASSES = len(label_encoder.classes_)


class ECGClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        self.model = ECGModel(input_dim=X_train.shape[1], num_classes=NUM_CLASSES)
        self.x_train, self.y_train = CLIENT_DATA[self.cid]

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train_model(
            self.model,
            self.x_train,
            self.y_train,
            epochs=LOCAL_EPOCHS,
            lr=LEARNING_RATE,
        )
        return get_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        acc, _ = evaluate_model(self.model, X_test, y_test)
        loss = 1.0 - acc
        return float(loss), len(X_test), {"accuracy": float(acc)}


def client_fn(cid: str):
    return ECGClient(cid)


class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    def evaluate(self, server_round, parameters):
        model = ECGModel(input_dim=X_train.shape[1], num_classes=NUM_CLASSES)
        set_parameters(model, parameters)
        acc, _ = evaluate_model(model, X_test, y_test)
        loss = 1.0 - acc
        return loss, {"accuracy": acc}


def run_federated():
    strategy = SaveMetricsStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    out_file = RESULTS_DIR / "federated_history.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(str(history))

    print(f"Saved federated history to: {out_file}")
