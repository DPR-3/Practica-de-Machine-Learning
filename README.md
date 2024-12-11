# Practica-de-Machine-Learning
El objetivo de esta práctica consiste en a partir de unos datos, extraer el mayor valor posible de los mismos para resolver una serie de problemáticas de negocio mediante las técnicas que hemos aprendido hasta ahora.

En cuanto a los directorios, archivos, nomenclaturas etc, serán muy similares a los de la práctica de Python. Se sugiere utilizar el código de esta como punto de partida.

Debe de crearse un paquete llamado models y en él tiene que haber:

Un script de Python con cada modelo necesario (ej svm_classifier.py, mlp_regressor.py, kmeans_clustering.py…) que contenga una clase con el modelo (ej: SVMClassifier, MLPRegressor…). Estas clases tendrá 5 métodos públicos y todos los métodos privados que sean necesarios (por ejemplo se puede crear un método privado llamado __preprocess_data() que sirva como método auxiliar para realizar las transformaciones previas requeridas en aquellos modelos que las necesiten):

El __init__ al que se le deberán pasar los parámetros que se consideren oportunos (puede que no haga falta pasarle ninguno).
Un método llamado fit que recibirá unos datos de entrada en formato "Data Frame" o “numpy array” y que entrenará el modelo, guardando los pesos y demás atributos que se consideren necesario guardar dentro del self de la clase.
Un método llamado predict que utilizará esta información guardada en la clase para hacer predicciones en base a los nuevos datos. Por ejemplo, si el modelo es de clasificación, este método debería devolver la clase o clases a la que corresponden los datos consultados. Debe ser capaz tanto de realizar inferencia de una sola observación como de un conjunto de ellas.
Un método llamado save que permita guardar en disco serializado en pickle los datos guardados dentro de la clase (pesos, estados de objetos preprocesadores…), que permitan en el futuro poder seguir utilizando el modelo sin reentrenarlo.
Un método llamado load que permita volver a cargar de disco el estado de la propia clase para poder volver a llamar a predict en una futura sesión sin tener que “reentrenar” el modelo.
Fuera de este paquete debe de haber 4 archivos más.

Un archivo llamado preprocess_data.py el cual realizará las transformaciones, limpiezas de datos y cambios requeridos comunes para todos los modelos. Las transformaciones específicas de cada modelo deberán realizarse dentro de su clase correspondiente, no en este fichero. Debe de recibir dos argumentos de entrada, uno con la ruta del fichero de datos “en crudo” y otro con la ruta donde dejar los datos ya limpios y procesados para los modelos.
Un archivo llamado train_models.py que importará dicho paquete, instanciará los modelos, los entrenará invocando el método "fit" de cada uno con el conjunto de datos y luego serializará a disco lo aprendido mediante el método "save". Tanto la ruta desde donde leer los datos de origen como las rutas a donde guardar los ficheros pickle de cada modelo se deberán obtener de un fichero de configuración en formato configparser al que llamaremos train.conf. La ruta de este fichero de configuración se le deberá pasar como argumento al script y se deberá utilizar la librería argparse para procesar dicho argumento. Se recomienda crear secciones dentro de dicho fichero de configuración.
Un archivo llamado inference_model.py que instanciará un modelo, leerá mediante el método "load" lo ya aprendido y realizará la inferencia de nuevos datos. Recibirá tres argumentos de entrada, y se deberá utilizar la librería argparse para procesar dichos argumentos. Estos argumentos serán el tipo de modelo que se debe utilizar para hacer inferencia, los pesos concretos a utilizar de dicho modelo y el fichero de datos de entrada sobre el cual queremos realizar la inferencia.
Un notebook de Jupyter, al que llamaremos exploratory_analysis.ipynb. Este notebook se deben cargar los datos, realizar un análisis exploratorio exhaustivo intentando sacar la mayor parte de información posible de los mismos y luego se deben probar con los datos todas las técnicas y modelos que apliquen a la resolución del problema vistos durante el módulo 3 (machine learning). Se debe comparar el rendimiento de cada técnica que esté orientada al mismo propósito y mediante este procedimiento elegir las que finalmente se industrializarán dentro del paquete models. Asimismo, no solo se deben comparar técnica a técnica, sino que una vez elegida una de ellas, se tienen que intentar encontrar los hiperparámetros óptimos y las transformaciones de los datos de entrada más adecuadas para obtener los mejores resultados.
Ahora vamos a hablar sobre el dataset utilizado y los problemas de negocio que queremos resolver. El dataset incluye varios meses de datos sobre los videos de tendencias diarias de YouTube. Los datos también incluyen un campo category_id, que varía entre regiones. Para recuperar las categorías de un video específico, hay que buscarlo en el JSON asociado. Se incluye uno de esos archivos para cada una de las regiones del conjunto de datos. El primer paso del pre-procesamiento deberá ser cruzar ambos archivos para cada región y sustituir los IDs en los CSV por las etiquetas que correspondan. Después, se deben juntar todos los datos enriquecidos en el paso anterior de todas las regiones en un solo CSV que incluya una nueva columna con el nombre de la región. Toda esta lógica deberá implementarse en preprocess_data.py.

Se deberán probar todos los modelos posibles para cada uno de los siguientes problemas de negocio, industrializando en el paquete solamente el mejor de ellos para cada uno de los casos, con los mejores hiperparámetros y transformaciones previas encontradas.

Clasificación:
Reto normal: Crear un clasificador que prediga la categoría del vídeo.
Reto opcional para subir nota: Crear un algoritmo de clasificación que intente averiguar lo mejor posible si los comentarios o los ratings estarán deshabilitados por parte del creador del vídeo en base al resto de la información. Al ser una clasificación altamente desbalanceada se valorará no solo obtener una alta precisión, sino también estudiar en una matriz de confusión si se sega excesivamente hacia la clase mayoritaria y en el caso de que ocurra intentar mitigar este hecho.
Regresión:
Reto normal: Intentar predecir el número de likes.
Reto opcional para subir nota: Intentar predecir la ratio de likes/dislikes de cada vídeo.
Clusterización: Intentar encontrar grupos en los vídeos mediante diversas técnicas de clusterización. Evaluar cómo se superpone cada una con respecto a las categorías de los vídeos. Utilizar las técnicas de reducción de dimensionalidad.
Recomendación: Crear un recomendador que dado un vídeo te recomiende vídeos parecidos.
 

Dataset:

https://drive.google.com/file/d/16rdg0eP4e5Db99sAXwAQfSOvgUatOgLj/view?usp=sharing
