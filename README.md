# Detección de Fraude en Transacciones Digitales – Trabajo Práctico Integrador de la Diplomatura en Ciencia de Datos y Análisis Avanzado

Este repositorio contiene el código y los recursos para el Trabajo Práctico Integrador enfocado en la detección de fraude en transacciones digitales mediante técnicas de Machine Learning y Deep Learning. El objetivo del proyecto es evaluar la capacidad predictiva de un conjunto de variables transaccionales, de comportamiento y de contexto para clasificar transacciones como fraudulentas o no fraudulentas, comparando distintas estrategias de modelado y validación.

## Contenido

- `TPintegrador_Grupo_4.ipynb`: notebook Jupyter con todo el análisis del proyecto, incluyendo análisis exploratorio de datos (EDA), tratamiento de valores faltantes, feature engineering, modelado, evaluación y comparación de estrategias.
- `data/Fraud Detection Dataset.csv`: dataset utilizado por el notebook.
- `README.md`: este archivo, con la descripción general del trabajo, requisitos y forma de ejecución.

## Objetivo del trabajo

El objetivo del presente trabajo es generar un modelo que permita detectar posibles transacciones fraudulentas realizadas en una plataforma de pago virtual para evitar que los consumos impacten de manera automática. Para ello se trabaja con el dataset de Kaggle, que contiene aproximadamente 51.000 transacciones y 4.000 usuarios únicos.

A lo largo del trabajo se evalúan dos estrategias principales:

1. **Estrategia 1 – Modelado a nivel transacción**: se utilizan directamente las variables disponibles en el dataset luego del preprocesamiento.
2. **Estrategia 2 – Modelado con feature engineering por usuario**: se construyen variables adicionales a partir del comportamiento histórico de cada usuario, con el objetivo de capturar cambios en el patrón de uso entre transacciones sucesivas.

## Variables del dataset

El conjunto de datos incluye, entre otras, las siguientes variables:

- `Transaction_ID`
- `User_ID`
- `Transaction_Amount`
- `Transaction_Type`
- `Time_of_Transaction`
- `Device_Used`
- `Location`
- `Previous_Fraudulent_Transactions`
- `Account_Age`
- `Number_of_Transactions_Last_24H`
- `Payment_Method`
- `Fraudulent`

## Metodología general

El flujo de trabajo desarrollado en el notebook incluye:

- Carga y exploración inicial del dataset.
- Evaluación del desbalance de clases.
- Análisis exploratorio de variables numéricas y categóricas.
- Imputación de valores faltantes:
  - variables categóricas con la categoría `Unknown`
  - variables numéricas con la mediana
- Codificación de variables categóricas mediante **one-hot encoding**.
- Construcción de variables derivadas de comportamiento por usuario para la segunda estrategia.
- Separación de los datos en entrenamiento y test, evitando la superposición de usuarios entre ambos subconjuntos.
- Escalado de variables cuando corresponde.
- Balanceo del conjunto de entrenamiento mediante **SMOTE**.
- Entrenamiento y comparación de múltiples modelos.
- Evaluación mediante métricas apropiadas para clases desbalanceadas.

## Modelos evaluados

En el trabajo se comparan distintos modelos supervisados, incluyendo:

- Regresión Logística
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Red neuronal densa (DNN)

La comparación de modelos permite analizar si el problema presenta relaciones lineales, reglas simples, interacciones no lineales o patrones más complejos. En este caso, el bajo desempeño general obtenido resulta útil para discutir la baja capacidad explicativa del dataset.

## Métricas de evaluación

Debido al desbalance entre clases, la evaluación no se basa únicamente en accuracy. Se reportan las siguientes métricas:

- **ROC-AUC**
- **PR-AUC**
- **F1-score**
- **Accuracy**
- **Recall**
- **Precision**

En el contexto del problema, las métricas más relevantes son **PR-AUC** y **Recall**, dado que permiten evaluar mejor la detección de la clase minoritaria (fraude).

## Requisitos

- Python 3.10 o superior.
- Jupyter Notebook o JupyterLab para ejecutar el archivo `.ipynb`.
- Librerías de Python:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `imbalanced-learn`
  - `xgboost`
  - `lightgbm`
  - `catboost`
  - `tensorflow`

## Instalación

1. Clona o descarga este repositorio.
2. Crea un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv env
   source env/bin/activate
   ```

3. Instala las dependencias necesarias:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost tensorflow jupyterlab notebook
   ```

## Ejecución del notebook

1. Verifica que el dataset exista en la ruta:

   ```bash
   data/Fraud\ Detection\ Dataset.csv
   ```

2. Desde la raíz del proyecto, abre el notebook con Jupyter Notebook, JupyterLab o Visual Studio Code:

   ```bash
   jupyter lab TPintegrador_Grupo_4.ipynb
   ```

   También puedes usar:

   ```bash
   jupyter notebook TPintegrador_Grupo_4.ipynb
   ```

3. Ejecuta las celdas en orden.

4. El notebook carga el dataset desde `data/Fraud Detection Dataset.csv` y compara dos estrategias:
   - Estrategia 1: partición train/test estratificada a nivel transacción.
   - Estrategia 2: partición por usuario para evitar que un mismo usuario aparezca en train y test.

5. Al finalizar, podrás comparar el desempeño de los distintos modelos y discutir los resultados en función de las características del dataset.

## Principales hallazgos del trabajo

Entre los hallazgos más relevantes se destacan:

- Existe un fuerte desbalance de clases (~5% fraude) que condiciona el problema.
- Las variables muestran distribuciones muy similares entre fraude y no fraude, sin capacidad clara de separación.
- Se observa baja correlación con la variable objetivo, indicando limitada capacidad explicativa.
- Todos los modelos presentan un desempeño cercano al azar, lo que sugiere baja señal predictiva en el dataset.
- La estrategia basada en comportamiento de usuario no mejora significativamente los resultados.
- El ajuste del threshold no permitió reducir los falsos positivos sin afectar otras métricas, reforzando la limitación del dataset.

En conjunto, la principal limitación está en los datos, no en los modelos.

## Posibles líneas futuras de mejora

- Integración de datos externos o contextuales (por ejemplo, geolocalización más precisa, características del dispositivo o redes de conexión entre usuarios).
- Evaluación de técnicas avanzadas de explicabilidad, que permitan comprender mejor las decisiones del modelo y detectar posibles fuentes de ruido o sesgo en los datos.

## Referencias

- Dataset: Kaggle – www.kaggle.com/datasets/ranjitmandal/fraud-detection-dataset-csv 
