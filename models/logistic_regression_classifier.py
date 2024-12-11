import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd


class LogisticRegressionClassifier:
    def __init__(self, C=1.0, max_iter=100, scale_features=False):
        self.model = LogisticRegression(C=C, max_iter=max_iter)
        self.scale_features = scale_features
        if scale_features:
            self.scaler = StandardScaler()

    def fit(self, X, y):
        # Validar los datos de entrada
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X debe ser un DataFrame y y debe ser una Serie de pandas.")

        # Escalar características si es necesario
        if self.scale_features:
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        # Ajustar el modelo
        self.model.fit(X, y)

    def predict(self, X):
        # Validar los datos de entrada
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame.")

        # Escalar características si es necesario
        if self.scale_features:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def save(self, filename):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")

    def load(self, filename):
        try:
            with open(filename, 'rb') as f:
                loaded_model = pickle.load(f)
                self.model = loaded_model.model
                self.scaler = loaded_model.scaler
                self.scale_features = loaded_model.scale_features
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
