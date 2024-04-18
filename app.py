from flask import Flask, request, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.regularizers import l2
import cv2
import numpy as np

app = Flask(__name__)

def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    return model

model = build_model()

dummy_input = np.zeros((1, 128, 128, 3))
model.predict(dummy_input)

model.load_weights('modelos/pesos_modelo.weights.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file and file.filename:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "Error: The provided file is not a valid image.", 400
        img = cv2.resize(img, (128, 128))
        img = preprocess_input(img)
        img = np.array(img).reshape(-1, 128, 128, 3)

        prediction = model.predict(img)
        classes = ['San_Juditas_Tadeo', 'San_Antonio_de_Padua', 'San_Benito', 'San_Francisco_de_Asis',
                   'San_Martin_Caballero', 'Santo_Nino_de_Atocha', 'Virgen_de_Guadalupe', 'Virgen_de_Juquila',
                   'Virgen_de_la_Candelaria', 'Virgen_de_los_Remedios']
        predicted_class = classes[np.argmax(prediction)]

        return predicted_class
    return 'No file provided', 400

if __name__ == '__main__':
    app.run(debug=True)
