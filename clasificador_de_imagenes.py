from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

modelo = MobileNetV2(weights='imagenet')

def clasificar_imagen(ruta_imagen):
    try:
        imagen = image.load_img(ruta_imagen, target_size=(224, 224))
        x = image.img_to_array(imagen)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        predicciones = modelo.predict(x)
        decodificadas = decode_predictions(predicciones, top=3)[0]
        return [(desc, float(conf)) for (_, desc, conf) in decodificadas]
    except Exception as error:
        print(f"[ERROR]: {error}")
        return [("Error interno al clasificar la imagen", 0.0)]
