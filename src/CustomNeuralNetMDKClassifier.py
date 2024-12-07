from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense


class CustomNeuralNetMDKClassifier:
    def __init__(
        self,
        num_classes=41,
        epochs=20,
        batch_size=32,
        verbose=0,
        layers=[128, 64],
        optimizer="adam",
    ):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.layers = layers
        self.optimizer = optimizer
        self.history_ = None  # Store history here
        self.model_ = None

    def build_model(self, input_dim):
        return KerasClassifier(
            build_fn=MDKMLClassifierv3,
            input_dim=input_dim,
            num_classes=self.num_classes,
            layers=self.layers,
            optimizer=self.optimizer,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

    def get_params(self, deep=True):
        return {
            "num_classes": self.num_classes,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
            "layers": self.layers,
            "optimizer": self.optimizer,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def fit(self, X, y):
        self.model_ = self.build_model(input_dim=X.shape[1])
        fit_result = self.model_.fit(X, y).history  # Capture history
        self.history_ = fit_result
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        return self.model_.score(X, y)


def MDKMLClassifierv3(input_dim, num_classes=41, layers=[128, 64], optimizer="adam"):
    model = Sequential()
    # Input layer + first hidden layer
    model.add(Dense(layers[0], input_dim=input_dim, activation="relu"))
    # Second hidden layer
    model.add(Dense(layers[1], activation="relu"))
    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
