import speech_recognition as sr
import pyttsx3

def voz_a_texto():
    reconocedor = sr.Recognizer()
    with sr.Microphone() as fuente:
        print("Habla ahora...")
        audio = reconocedor.listen(fuente)
        try:
            return reconocedor.recognize_google(audio, language="es-ES")
        except sr.UnknownValueError:
            return "No se entendió"
        except sr.RequestError:
            return "Error con el servicio"

def texto_a_voz(texto):
    motor = pyttsx3.init()
    motor.say(texto)
    motor.runAndWait()
