import cv2
import numpy as np

def Apply_TopHat(image, kernel_size):
    # Create a rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Apply grayscale open operation
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Calculate the top hat by subtracting the opened image from the original image
    top_hat = cv2.subtract(image, opened)
    
    return top_hat

# Carga la imagen (reemplaza 'imagen.jpg' con la ruta de tu imagen)
image = cv2.imread('./Selection/Agricam_18F.JPG', cv2.IMREAD_GRAYSCALE)

# Tamaño del kernel para el filtro Top Hat
kernel_size = (10, 10)

# Aplica el filtro Top Hat
top_hat_result = Apply_TopHat(image, kernel_size)

# Muestra la imagen original y el resultado del filtro Top Hat
cv2.imshow('Original', image)
cv2.imshow('Top Hat Result', top_hat_result)
cv2.waitKey(0)
cv2.destroyAllWindows()





