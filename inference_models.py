import argparse
import pandas as pd
from models.random_forest_model import RandomForestModel
def make_inference(model_type, model_path, data_file, vectorizer_path=None):
    if model_type == 'knn':
        from models.knn_classifier import KNNClassifier
        model = KNNClassifier()
    elif model_type == 'rf':
        from models.random_forest_model import RandomForestModel
        model = RandomForestModel()
    else:
        raise ValueError("Tipo de modelo no válido")

    # Cargar el modelo
    model.load(model_path)

    # Cargar datos para inferencia
    data = pd.read_csv(data_file)

    # Verificar columnas requeridas
    required_columns = ['likes', 'views', 'dislikes', 'comment_count']
    print(f"Columnas en el conjunto de datos: {data.columns}")
    print(f"Columnas usadas para la predicción: {required_columns}")

    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Columna faltante: {col}")

    # Preprocesar datos
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Preparar datos para inferencia
    X = data[required_columns]

    # Realizar la predicción
    predictions = model.predict(X)

    # Mostrar resultados
    print(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", help="Tipo de modelo (knn, logreg, rf, nb)")
    parser.add_argument("model_path", help="Ruta al modelo guardado")
    parser.add_argument("data_file", help="Ruta al archivo de datos para inferencia")
    parser.add_argument("--vectorizer_path", help="Ruta al vectorizador (solo para Naive Bayes)", default=None)
    args = parser.parse_args()

    make_inference(args.model_type, args.model_path, args.data_file, args.vectorizer_path)



