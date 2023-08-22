"""
Created on Mon Aug 21 12:11:33 2023
@author: Josué Aldana

PROCEDIMIENTO:

Conocer la resolución de imágenes y videos.

"""
import cv2

def obtener_resolucion_archivo(ruta_archivo):
    if ruta_archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif','JPG')):
        imagen = cv2.imread(ruta_archivo)
        if imagen is not None:
            alto, ancho = imagen.shape[:2]
            return f"Es una imagen con resolución: {ancho} x {alto}"
        else:
            return "No se pudo cargar la imagen"
    elif ruta_archivo.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(ruta_archivo)
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return f"Es un video con resolución: {ancho} x {alto}"
    else:
        return "Formato de archivo no compatible"

informacion_img = obtener_resolucion_archivo('./Sample/Foto_Muestra.JPG')
informacion_video = obtener_resolucion_archivo('./Sample/Video_Muestra.mp4')
print(informacion_img)
print(informacion_video)