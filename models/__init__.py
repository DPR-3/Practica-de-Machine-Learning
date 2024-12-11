import argparse
import configparser
import pandas as pd
import os
from models.knn_classifier import KNNClassifier
from models.logistic_regression_classifier import LogisticRegressionClassifier
from models.random_forest_model import RandomForestModel
from models.naive_bayes_classifier import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def train_models(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    base_path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(base_path, config['paths']['processed_data']))

    # Entrenar y guardar modelos
    knn = KNNModel()
    knn.fit(data[['likes', 'views', 'comment_count']], data['target'])
    knn.save(os.path.join(base_path, config['models']['knn_model']))

    logreg = LogisticRegressionClassifier()
    logreg.fit(data[['likes', 'views', 'comment_count']], data['target'])
    logreg.save(os.path.join(base_path, config['models']['logreg_model']))

    rf = RandomForestModel()
    rf.fit(data[['likes', 'views', 'comment_count']], data['target'])
    rf.save(os.path.join(base_path, config['models']['rf_model']))

    vectorizer = CountVectorizer()
    X_nb = vectorizer.fit_transform(data['title'])

    nb = NaiveBayesClassifier()
    nb.fit(X_nb, data['target'])
    nb.save(os.path.join(base_path, config['models']['nb_model']))

    with open(os.path.join(base_path, config['models']['nb_vectorizer']), 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Ruta al archivo de configuración")
    args = parser.parse_args()

    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_file)
    train_models(config_file_path)
