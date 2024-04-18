import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

classes = ['San_Juditas_Tadeo', 'San_Antonio_de_Padua', 'San_Benito', 'San_Francisco_de_Asis', 'San_Martin_Caballero', 'Santo_Nino_de_Atocha', 'Virgen_de_Guadalupe', 'Virgen_de_Juquila', 'Virgen_de_la_Candelaria', 'Virgen_de_los_Remedios']
num_classes = len(classes)
img_rows, img_cols = 128, 128

def load_data():
    data = []
    target = []
    
    for index, clase in enumerate(classes):
        folder_path = os.path.join('Entrenamiento', clase)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (img_rows, img_cols))
            image = preprocess_input(image)
            data.append(image)
            target.append(index)
    
    data = np.array(data)
    target = np.array(target)
    new_target = to_categorical(target, num_classes)
    
    return data, new_target

data, target = load_data()
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=np.argmax(target, axis=1))

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))
base_model.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

checkpoint = ModelCheckpoint('modelos/mejor_modelo.weights.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

if not os.path.exists('modelos'):
    os.makedirs('modelos')
model.save_weights('modelos/pesos_modelo.weights.h5')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

if not os.path.exists('graficas'):
    os.makedirs('graficas')

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.savefig('graficas/matriz_confusion.png')
plt.close()

plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Historial de pérdida')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend()
plt.savefig('graficas/historial_perdida.png')
plt.close()

plt.figure(figsize=(10, 8))
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Historial de precisión')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend()
plt.savefig('graficas/historial_precision.png')
plt.close()
