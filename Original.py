import cv2
import numpy as np
import math

def calculate_area_per_pixel_factor(resolution_horizontal, resolution_vertical, focal_length, drone_height):
    # Convertir el ángulo de apertura a radianes
    angle_of_view_rad = math.radians(focal_length)
    # Calcular el tamaño del píxel en metros en ambas dimensiones
    pixel_size_horizontal_m = 2 * drone_height * math.tan(angle_of_view_rad / 2) / resolution_horizontal
    pixel_size_vertical_m = 2 * drone_height * math.tan(angle_of_view_rad / 2) / resolution_vertical
    # Calcular el factor de conversión de área por píxel (metros cuadrados/píxel)
    factor = pixel_size_horizontal_m * pixel_size_vertical_m
    return factor

def Image_Reading(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load the image.")
        return
    else:
        return image

def Apply_Green_Highlight(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range of green color in HSV
    lower_green = np.array([24, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Create a mask for green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Create a black image of the same size as input
    highlight_image = np.zeros_like(image)
    
    # Set the green region in the highlight image
    highlight_image[green_mask > 0] = [0, 255, 0]  # Highlight in green
    
    # Combine the highlight image and the original image
    highlighted_image = cv2.addWeighted(image, 0.7, highlight_image, 0.3, 0)
    
    return highlighted_image, green_mask

if __name__ == "__main__":
    image_path = './Selection/Agricam_01F.JPG'

    original_image = Image_Reading(image_path)
    resized_image = cv2.resize(original_image, (800, 600))

    green_highlighted_image, green_mask = Apply_Green_Highlight(resized_image)

    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set a threshold for minimum contour area to segment by volume
    min_contour_area = 500  # Adjust this value according to your needs

    # Create a black mask to draw the segmented objects
    segment_mask = np.zeros_like(green_mask)

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            cv2.drawContours(segment_mask, [contour], -1, 255, -1)

    # Multiply segment_mask with the original image to visualize segmentation result
    segmented_result = cv2.bitwise_and(resized_image, resized_image, mask=segment_mask)

    # Calculate the percentage of white and black pixels in the segment mask
    total_pixels = segment_mask.shape[0] * segment_mask.shape[1] 
    white_pixels = np.sum(segment_mask == 255)
    black_pixels = total_pixels - white_pixels
    shades_percentage = (white_pixels / total_pixels) * 100
    no_shades_percentage = (black_pixels / total_pixels) * 100
    
    # Value of shades in square meters
    # Datos proporcionados
    resolution_horizontal = 3840  # Resolución horizontal de la imagen en píxeles
    resolution_vertical = 2160    # Resolución vertical de la imagen en píxeles
    focal_length_deg = 155         # Ángulo de apertura focal en grados
    drone_height_m = 28          # Altura del dron sobre el terreno en metros
    factor = calculate_area_per_pixel_factor(resolution_horizontal, resolution_vertical, focal_length_deg, drone_height_m)
    
    shades_decimal= white_pixels * factor
    
    print(f"Shades[%]: {shades_percentage:.2f}%")
    print(f"Shades[\u33A1]: {shades_decimal:.2f}")

    # Display the images
    cv2.imshow('Original Image', resized_image)
    cv2.imshow('Green Highlighted Image', green_highlighted_image)
    cv2.imshow('Green Mask', green_mask)
    cv2.imshow('Segment Mask', segment_mask)
    cv2.imshow('Segment result', segmented_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()