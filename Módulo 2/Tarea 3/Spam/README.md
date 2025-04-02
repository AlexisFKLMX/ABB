# Clasificador de SPAM

## 1. Introducción

Este proyecto consiste en desarrollar un clasificador de correos electrónicos que determina si un mensaje es **spam** o **no spam** utilizando el teorema de Bayes. Se utiliza un clasificador Naive Bayes (MultinomialNB) junto con la técnica de TF-IDF para la extracción de características. Además, se ha implementado una interfaz gráfica con Tkinter que permite al usuario redactar o cargar correos electrónicos y clasificar su contenido en tiempo real.

## 2. Requisitos

- **Lenguaje:** Python 3.x
- **Librerías:**
  - **Pandas:** Para manejo y procesamiento de datos.
  - **Scikit-learn:** Para la extracción de características (TF-IDF) y el entrenamiento del modelo Naive Bayes.
  - **NLTK:** Para la manipulación del texto y eliminación de stopwords.
  - **Tkinter:** Para la creación de la interfaz gráfica.
- **Dataset:** Archivo CSV (`spam_assassin.csv`) proveniente del SpamAssassin Public Corpus, que contiene dos columnas:
  - `text`: El contenido del correo.
  - `target`: La etiqueta (1 para spam y 0 para no spam).

## 3. Descripción del Dataset

El dataset está conformado por correos electrónicos etiquetados. Los datos fueron obtenidos y limpiados de los conjuntos de correos:
- **Spam:** Combina los grupos *spam* y *spam_2*.
- **No spam (ham):** Combina *easy_ham* y *easy_ham_2*.

Se han eliminado los correos duplicados y los que no estaban codificados en UTF-8.

## 4. Preprocesamiento

El preprocesamiento del texto incluye los siguientes pasos:

- **Conversión a minúsculas:** Se asegura la uniformidad del texto.
- **Eliminación de caracteres especiales:** Se eliminan caracteres que no sean letras, números o espacios.
- **Tokenización:** Se divide el texto en palabras.
- **Eliminación de stopwords:** Se remueven palabras comunes en inglés (como "the", "is", "and", etc.) que no aportan significado significativo al modelo.

La función `preprocess_text` implementa estos pasos y devuelve el texto limpio listo para la vectorización.

## 5. Extracción de Características y Entrenamiento del Modelo

Para extraer las características del texto se utiliza el `TfidfVectorizer` de scikit-learn, el cual convierte el texto en una matriz TF-IDF:

- **TF (Term Frequency):** Cuenta la frecuencia de cada palabra en el correo.
- **IDF (Inverse Document Frequency):** Mide la importancia de cada palabra en el conjunto de datos.
- **TF-IDF:** Combina ambas medidas para ponderar las palabras en función de su relevancia.

El modelo se entrena usando un clasificador Naive Bayes (MultinomialNB). Se divide el dataset en conjuntos de entrenamiento y prueba para evaluar la precisión del modelo.

## 6. Interfaz Gráfica (GUI) con Tkinter

La interfaz gráfica permite al usuario:

- **Redactar o cargar correos electrónicos:**  
  Se incluye un área de texto para ingresar el contenido y un botón para cargar archivos de texto.
  
- **Clasificar el correo:**  
  Un botón "Clasificar" procesa el texto, lo preprocesa, lo vectoriza y muestra el resultado (SPAM o No SPAM) en un cuadro de diálogo.

### Funcionalidades Clave

- **`load_text_file`:**  
  Abre un cuadro de diálogo para seleccionar un archivo de texto (.txt) y carga su contenido en el área de texto.

- **`classify_email`:**  
  Toma el texto ingresado, lo preprocesa, transforma con TF-IDF y utiliza el modelo entrenado para predecir la etiqueta del correo.

## 7. Cómo Usar el Programa

1. **Instalación de dependencias:**  
   Asegúrate de tener instaladas las siguientes librerías:
   
       pip install pandas scikit-learn nltk
   
   Tkinter generalmente viene incluido con Python, pero si no lo tienes, consulta la documentación de tu sistema.

2. **Preparación del dataset:**  
   Coloca el archivo `spam_assassin.csv` en el mismo directorio que el código.

3. **Ejecución del código:**  
   Ejecuta el script de Python. Durante la ejecución se descargarán las stopwords de NLTK si es la primera vez.

4. **Uso de la GUI:**  
   - Escribe o carga el texto del correo mediante el botón **"Cargar Archivo"** (En el repositorio hay varios archivos de prueba).
   - Presiona el botón **"Clasificar"** para ver el resultado en un cuadro de diálogo.

## 8. Explicación del Código

### 8.1. Importaciones y Configuración Inicial

Se importan las librerías necesarias y se descargan las stopwords desde NLTK. Además, se define el conjunto de stopwords en inglés para el preprocesamiento.

### 8.2. Función `preprocess_text`

Esta función toma un string y realiza:
- Conversión a minúsculas.
- Eliminación de caracteres especiales mediante expresiones regulares.
- Tokenización y eliminación de palabras vacías.
- Retorno del texto limpio.

### 8.3. Carga y Preparación del Dataset

Se carga el dataset desde el archivo CSV, se eliminan duplicados y se aplica el preprocesamiento para generar una nueva columna `clean_text`.

### 8.4. División del Dataset y Vectorización

Se divide el dataset en conjuntos de entrenamiento y prueba. El `TfidfVectorizer` se ajusta a los datos de entrenamiento y transforma ambos conjuntos en matrices de características.

### 8.5. Entrenamiento y Evaluación del Modelo

El modelo MultinomialNB se entrena con los datos vectorizados. Se evalúa el desempeño del modelo calculando la precisión con `accuracy_score`, imprimiéndose el resultado en consola.

### 8.6. Implementación de la Interfaz Gráfica

Se utiliza Tkinter para crear la ventana principal con:
- **Área de texto:** Para ingresar o cargar correos.
- **Botón "Clasificar":** Para ejecutar la función `classify_email` y mostrar la clasificación.
- **Botón "Cargar Archivo":** Para abrir un cuadro de diálogo que permite cargar archivos de texto.

### 8.7. Funciones de la GUI

- **`load_text_file`:**  
  Permite seleccionar un archivo `.txt` y cargar su contenido en el área de texto.

- **`classify_email`:**  
  Toma el contenido del área de texto, lo preprocesa, lo vectoriza y predice la clase del correo, mostrando el resultado en un mensaje.

## 9. Conclusión

Este proyecto integra el procesamiento de lenguaje natural, la extracción de características y el aprendizaje automático en un flujo de trabajo que culmina en una interfaz de usuario amigable para clasificar correos electrónicos. La modularidad del código permite futuras mejoras, como el ajuste de parámetros del modelo o la incorporación de nuevos métodos de preprocesamiento.
