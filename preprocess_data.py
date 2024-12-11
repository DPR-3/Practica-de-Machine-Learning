import pandas as pd
import json
import argparse
import os
import re

def preprocess_data(raw_data_path, processed_data_path):
    all_data = pd.DataFrame()

    # Obtener todas las regiones disponibles (usando expresiones regulares)
    regions = []
    for filename in os.listdir(raw_data_path):
        if filename.endswith('.csv'):
            match = re.search(r'region_([A-Z]{2})\.csv$', filename)  # Busca "region_" seguido de dos letras mayúsculas y ".csv" al final
            if match:
                regions.append(match.group(1))

    for region in regions:
        # Leer datos crudos y JSON de categorías
        csv_path = os.path.join(raw_data_path, f"region_{region}.csv")
        json_path = os.path.join(raw_data_path, f"region_{region}_category_id.json")

        if os.path.exists(csv_path) and os.path.exists(json_path):
            df = pd.read_csv(csv_path)
            with open(json_path) as f:
                categories = json.load(f)

            # Crear un diccionario para mapear IDs a etiquetas
            category_mapping = {item['id']: item['snippet']['title'] for item in categories['items']}

            # Reemplazar IDs por etiquetas y añadir columna de región
            df['category'] = df['category_id'].map(category_mapping)
            df['region'] = region

            # Verificar que 'tags' existe antes de procesarla
            if 'tags' in df.columns:
                df['tags'] = df['tags'].str.replace('[', '').str.replace(']', '').str.replace("'", "").str.split(',')
            else:
                print(f"Advertencia: La columna 'tags' no se encontró en los datos de la región {region}")

            all_data = pd.concat([all_data, df])
        else:
            print(f"Advertencia: No se encontró el archivo CSV o JSON para la región {region}")

    # Guardar DataFrame procesado sin sobrescribir el archivo existente
    print(f"Guardando datos procesados en {processed_data_path}")
    all_data.to_csv(processed_data_path, mode='a', header=not os.path.exists(processed_data_path), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_path", help="Ruta a los datos crudos (relativa al proyecto)")
    parser.add_argument("processed_data_path", help="Ruta para guardar los datos procesados (relativa al proyecto)")
    args = parser.parse_args()

    # Obtener la ruta base del proyecto
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Construir rutas completas usando os.path.join y la ruta base
    raw_data_path = os.path.join(base_path, args.raw_data_path)
    processed_data_path = os.path.join(base_path, args.processed_data_path)

    # Mensajes de depuración para verificar las rutas
    print(f"Procesando datos de {raw_data_path}")
    print(f"Guardando datos procesados en {processed_data_path}")

    preprocess_data(raw_data_path, processed_data_path)

