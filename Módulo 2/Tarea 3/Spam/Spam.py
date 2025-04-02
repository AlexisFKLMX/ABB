from tkinter import filedialog
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer #scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import scrolledtext, messagebox

# Descargar stopwords de NLTK.
nltk.download('stopwords')

# Definir el conjunto de stopwords en inglés.
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesa el texto eliminando caracteres especiales, 
    convirtiendo a minúsculas, removiendo stopwords y tokenizando.
    """
    # Convertir a minúsculas.
    text = text.lower()
    # Eliminar caracteres especiales (manteniendo letras, números y espacios).
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Tokenizar el texto.
    tokens = text.split()
    # Remover stopwords.
    tokens = [word for word in tokens if word not in stop_words]
    # Reunir los tokens en un string.
    return ' '.join(tokens)

# Cargar el dataset.
df = pd.read_csv('spam_assassin.csv')

# Eliminar correos duplicados.
df.drop_duplicates(inplace=True)

# Aplicar preprocesamiento al texto.
df['clean_text'] = df['text'].apply(preprocess_text)

# Separar los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['target'], test_size=0.2, random_state=42)

# Extracción de características: vectorización TF-IDF.
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Entrenar el modelo Naive Bayes.
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluar el modelo en el conjunto de prueba.
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

def load_text_file():
    """
    Abre un cuadro de diálogo para seleccionar un archivo de texto (.txt),
    carga su contenido y lo inserta en el área de texto.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        # Limpiar el área de texto y luego insertar el contenido.
        text_area.delete("1.0", END)
        text_area.insert("1.0", content)

def classify_email():
    """
    Función para clasificar el correo ingresado en la interfaz.
    Preprocesa el texto, lo vectoriza y utiliza el modelo para predecir si es spam.
    """
    # Obtener el texto del área de entrada.
    email_text = text_area.get("1.0", END).strip()
    if email_text == "":
        messagebox.showwarning("Advertencia", "Por favor ingresa el texto del correo.")
        return
    # Preprocesar el texto ingresado.
    email_clean = preprocess_text(email_text)
    # Transformar el texto usando el vectorizador TF-IDF.
    email_vector = tfidf_vectorizer.transform([email_clean])
    # Predecir la clase: 1 = spam, 0 = no spam.
    prediction = model.predict(email_vector)[0]
    result = "SPAM" if prediction == 1 else "No SPAM"
    # Mostrar el resultado en un mensaje.
    messagebox.showinfo("Resultado", f"El correo se clasificó como: {result}")

# Configuración de la interfaz gráfica con Tkinter.
root = Tk()
root.title("Clasificador de Spam")

# Etiqueta de instrucciones.
label = Label(root, text="Ingresa el texto del correo electrónico:")
label.pack(pady=5)

# Área de texto con barra de desplazamiento.
text_area = scrolledtext.ScrolledText(root, wrap=WORD, width=60, height=15)
text_area.pack(padx=10, pady=10)

# Botón para clasificar el correo.
classify_button = Button(root, text="Clasificar", command=classify_email)
classify_button.pack(pady=10)

# Botón para cargar archivo
load_button = Button(root, text="Cargar Archivo", command=load_text_file)
load_button.pack(pady=5)

# Iniciar el bucle principal de la aplicación.
root.mainloop()
