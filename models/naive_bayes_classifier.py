import pickle
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)

    def fit(self, X, y):
        # Validar los datos de entrada
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X debe ser un DataFrame y y debe ser una Serie de pandas.")

        # Ajustar el modelo
        self.model.fit(X, y)

    def predict(self, X):
        # Validar los datos de entrada
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame.")

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
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
