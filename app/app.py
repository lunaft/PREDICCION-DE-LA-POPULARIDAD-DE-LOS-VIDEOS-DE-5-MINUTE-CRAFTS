from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Cargar el tokenizador desde el nuevo path.
tokenizer_path = '/Users/lunaflorestorres/Desktop/Flask 5-Minute Crafst/app/tokenizer.pickle'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Cargar el modelo desde el nuevo path.
model_path = '/Users/lunaflorestorres/Desktop/Flask 5-Minute Crafst/app/my_model.h5'
model = load_model(model_path)

# Ajusta max_length al valor utilizado durante el entrenamiento de tu modelo.
max_length = 19

@app.route('/') 
def index():
    # Renderiza la página de inicio.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el título del video del formulario
    video_title = request.form['video_title']
    
    # Preprocesar el título del video (convertir a minúsculas)
    video_title = video_title.lower()
    
    # Tokenizar el texto de entrada y realizar el padding
    seq = tokenizer.texts_to_sequences([video_title])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')
    
    # Realizar la predicción
    prediction = model.predict(padded_seq)
    prediction_label = np.argmax(prediction, axis=1)

    # Ajustar las categorías para reflejar la estructura de 4 grupos
    categories = ["Mal vídeo (Menos de 1.000.000 visualizaciones)", "Vídeo normal (1.000.000 - 10.000.000 visualizaciones)","Supervídeo (Más de 10.000.000 visualizaciones)"]
    predicted_category = categories[prediction_label[0]]
    
    # Renderizar la misma página index.html con la predicción
    return render_template('index.html', prediction=predicted_category)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
