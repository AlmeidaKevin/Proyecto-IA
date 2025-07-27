import tkinter as tk
from tkinter import filedialog, messagebox
from utilidades_del_modelo import cargar_datos, crear_modelo, entrenar_modelo, graficar_metricas
from clasificador_de_imagenes import clasificar_imagen
from modulo_de_voz import voz_a_texto, texto_a_voz

modelo_entrenado = None

def entrenar():
    global modelo_entrenado
    (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = cargar_datos()
    modelo_entrenado = crear_modelo()
    modelo_entrenado, historial = entrenar_modelo(modelo_entrenado, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)
    modelo_entrenado.save("modelo_mnist.h5")
    graficar_metricas(historial)
    messagebox.showinfo("Entrenamiento", "Modelo entrenado y guardado")

def clasificar():
    ruta_archivo = filedialog.askopenfilename()
    if not ruta_archivo:
        return

    resultados = clasificar_imagen(ruta_archivo)
    if resultados[0][0].lower().startswith("error"):
        messagebox.showerror("Error", resultados[0][0])
    else:
        mensaje = "\n".join([f"{etiqueta}: {round(conf * 100, 2)}%" for (etiqueta, conf) in resultados])
        messagebox.showinfo("Clasificación", f"Resultados:\n{mensaje}")

def voz_a_texto_interfaz():
    resultado = voz_a_texto()
    messagebox.showinfo("Reconocimiento de Voz", f"Dijiste: {resultado}")

def texto_a_voz_interfaz():
    texto = entrada.get()
    if texto:
        texto_a_voz(texto)
    else:
        messagebox.showerror("Error", "Escribe algo primero")

ventana = tk.Tk()
ventana.title("Proyecto IA - Clasificador y Voz")

tk.Button(ventana, text="Entrenar Modelo", command=entrenar).pack(pady=5)
tk.Button(ventana, text="Clasificar Imagen", command=clasificar).pack(pady=5)
tk.Button(ventana, text="Hablar (voz a texto)", command=voz_a_texto_interfaz).pack(pady=5)
entrada = tk.Entry(ventana)
entrada.pack(pady=5)
tk.Button(ventana, text="Leer Texto (texto a voz)", command=texto_a_voz_interfaz).pack(pady=5)

ventana.mainloop()
