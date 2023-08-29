import cv2
import numpy as np
import random

def filter_objects_by_area(binary_image, min_contour_area):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an image to store the selected objects
    selected_objects_image = np.zeros_like(binary_image)

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            cv2.drawContours(selected_objects_image, [contour], -1, 255, -1)

    return selected_objects_image


# Crear una imagen en blanco
width, height = 400, 300
image = np.zeros((height, width), dtype=np.uint8)

# Dibujar objetos aleatorios en la imagen
num_objects = random.randint(5, 10)
for _ in range(num_objects):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    w = random.randint(20, 80)
    h = random.randint(20, 80)
    cv2.rectangle(image, (x, y), (x + w, y + h), 255, -1)


selected_objects_image=filter_objects_by_area(image,2000)
# Mostrar las imágenes
cv2.imshow('Original Image', image)
cv2.imshow('Selected Objects', selected_objects_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


