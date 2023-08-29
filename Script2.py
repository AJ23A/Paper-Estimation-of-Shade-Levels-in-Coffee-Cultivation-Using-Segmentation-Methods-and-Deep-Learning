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
    lower_green = np.array([24, 40, 35])
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

def Apply_TopHat(image, kernel_size):
    # Crear un kernel circular para el filtro top hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Aplicar la operación de "top hat"
    top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    return top_hat
    
def Apply_OpeningAndClosing(image, kernel_size):
    # Create a rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Apply the opening operation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Apply the closing operation
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    
    
    return closed_image

def First_Stage(resized_image):
    green_highlighted_image, green_mask = Apply_Green_Highlight(resized_image)
    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Set a threshold for minimum contour area to segment by volume
    min_contour_area = 100  # Adjust this value according to your needs
    # Create a black mask to draw the segmented objects
    segment_mask = np.zeros_like(green_mask)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            cv2.drawContours(segment_mask, [contour], -1, 255, -1)
    # Multiply segment_mask with the original image to visualize segmentation result
    segmented_result = cv2.bitwise_and(resized_image, resized_image, mask=segment_mask)
    return segmented_result

def Second_Stage(segmented_result):
    # Apply the top hat filter
    top_hat_image = Apply_TopHat(segmented_result,(10, 10))
    
    # Apply opening and closing with a kernel size of (2, 2)
    opening_closing_image = Apply_OpeningAndClosing(top_hat_image, 10)
    
    # Apply green highlight to the opening and closing image
    green_highlighted_image2, green_mask2 = Apply_Green_Highlight(opening_closing_image)

    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set a threshold for minimum contour area to segment by volume
    min_contour_area = 500 #djust this value according to your needs

    # Create a black mask to draw the segmented objects
    segment_mask2 = np.zeros_like(green_mask2)

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            cv2.drawContours(segment_mask2, [contour], -1, 255, -1)

    # Multiply segment_mask with the original image to visualize segmentation result
    segmented_result2 = cv2.bitwise_and(segmented_result, segmented_result, mask=segment_mask2)
    
    return segment_mask2,segmented_result2,top_hat_image,opening_closing_image

def Percentage_shades( segment_mask2,segmented_result2,top_hat_image,opening_closing_image):
    total_pixels = segment_mask2.shape[0] * segment_mask2.shape[1] 
    white_pixels = np.sum(segment_mask2 == 255)
    shades_percentage = (white_pixels / total_pixels) * 100
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
        
if __name__ == "__main__":

    #Reading Image
    original_image = Image_Reading('./Selection/Agricam_18F.JPG')
    
    #Resize
    resized_image = cv2.resize(original_image, (800, 600))
    
    #First stage
    #segmented_result= First_Stage(resized_image)
    
    #Second Stage
    segment_mask2,segmented_result2,top_hat_image,opening_closing_image=Second_Stage(resized_image)  
    
    # Calculate the percentage of white and black pixels in the segment mask
    Percentage_shades( segment_mask2,segmented_result2,top_hat_image,opening_closing_image)

    # Display the images
    cv2.imshow('Original Image', resized_image)
    #cv2.imshow('Segmented by color', segmented_result)
    cv2.imshow('Top Hat', top_hat_image)   
    cv2.imshow('opening closing - Image', opening_closing_image)  
    cv2.imshow('Segment mask', segment_mask2)  
    cv2.imshow('Segment result2', segmented_result2)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()