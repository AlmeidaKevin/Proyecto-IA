from flask import Flask, request, jsonify
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import os

aplicacion = Flask(__name__)
modelo = MobileNetV2(weights='imagenet')

def clasificar_imagen(ruta_imagen):
    imagen = image.load_img(ruta_imagen, target_size=(224, 224))
    x = image.img_to_array(imagen)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predicciones = modelo.predict(x)
    decodificadas = decode_predictions(predicciones, top=3)[0]
    return [(desc, float(conf)) for (_, desc, conf) in decodificadas]

@aplicacion.route('/predecir', methods=['POST'])
def predecir():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró ninguna imagen'}), 400

    archivo_imagen = request.files['image']
    ruta_guardada = f"temp_{archivo_imagen.filename}"
    archivo_imagen.save(ruta_guardada)

    try:
        resultados = clasificar_imagen(ruta_guardada)
        os.remove(ruta_guardada)
        return jsonify({
            'predicciones': [
                {'etiqueta': desc, 'confianza': round(conf * 100, 2)}
                for desc, conf in resultados
            ]
        })
    except Exception as error:
        return jsonify({'error': str(error)}), 500

if __name__ == '__main__':
    aplicacion.run(debug=True)
