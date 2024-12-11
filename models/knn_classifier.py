import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


class KNNModel(BaseEstimator):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.columns_to_drop = ['video_id', 'trending_date', 'thumbnail_link', 'comments_disabled', 'ratings_disabled',
                                'video_error_or_removed', 'description']
        self.numeric_features = ['views', 'likes', 'comment_count']
        self.features_ = None

    def preprocess_data(self, X, fit=False):
        # Eliminar columnas no necesarias
        X = X.drop(self.columns_to_drop, axis=1, errors='ignore')

        # Verificar si 'publish_time' está en el DataFrame
        if 'publish_time' in X.columns:
            X['publish_time'] = pd.to_datetime(X['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
            X['publish_hour'] = X['publish_time'].dt.hour
            X['publish_day'] = X['publish_time'].dt.dayofweek
            X['publish_month'] = X['publish_time'].dt.month
            X = X.drop('publish_time', axis=1)
        else:
            print("La columna 'publish_time' no está presente en los datos")

        # Procesar la columna 'tags'
        if 'tags' in X.columns:
            X['tags'] = X['tags'].str.replace('[', '').str.replace(']', '').str.replace("'", "").str.split(',')
            X['tags_count'] = X['tags'].apply(len)
            X = X.drop('tags', axis=1)
        else:
            print("La columna 'tags' no está presente en los datos. Se omite el procesamiento de 'tags'.")

        # Verificar si 'category_id' y 'channel_title' están en el DataFrame
        if 'category_id' in X.columns and 'channel_title' in X.columns:
            if fit:
                X['category_id'] = self.label_encoder.fit_transform(X['category_id'])
                X['channel_title'] = self.label_encoder.fit_transform(X['channel_title'])
            else:
                X['category_id'] = self.label_encoder.transform(X['category_id'])
                X['channel_title'] = self.label_encoder.transform(X['channel_title'])
        else:
            print("Las columnas 'category_id' o 'channel_title' no están presentes en los datos.")

        # Verificar si 'comment_count' y otras columnas numéricas están presentes
        for feature in self.numeric_features:
            if feature not in X.columns:
                print(f"La columna '{feature}' no está presente en los datos.")

        # Crear características adicionales
        if all(feature in X.columns for feature in self.numeric_features):
            if 'dislikes' in X.columns:
                X['likes_ratio'] = X['likes'] / (X['likes'] + X['dislikes'])
            else:
                X['likes_ratio'] = X['likes'] / X['likes']  # Dividir entre sí mismo si 'dislikes' no está
            X['engagement_ratio'] = (X['likes'] + X['dislikes'] + X['comment_count']) / X['views']

        # Aplicar escalado a las características numéricas
        if fit and all(feature in X.columns for feature in self.numeric_features):
            X[self.numeric_features] = self.scaler.fit_transform(X[self.numeric_features])
        elif not fit and all(feature in X.columns for feature in self.numeric_features):
            X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])

        self.features_ = X.columns.tolist()  # Store the features used for training
        return X

    def fit(self, X, y):
        X_processed = self.preprocess_data(X, fit=True)
        self.model.fit(X_processed, y)
        return self

    def predict(self, X):
        if self.features_ is None:
            raise ValueError("No se han definido características. Asegúrese de que el modelo ha sido ajustado.")

        X_processed = self.preprocess_data(X, fit=False)
        return self.model.predict(X_processed)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
