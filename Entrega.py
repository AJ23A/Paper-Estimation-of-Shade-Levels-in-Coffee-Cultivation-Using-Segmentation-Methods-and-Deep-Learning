import cv2
import numpy as np
import math
import os

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

def Apply_TopHat(image):
    # Define the kernel size for top hat
    kernel_size = (10, 10)

    # Create a rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Apply the grayscale open operation as top hat
    top_hat = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return top_hat

def Apply_ClosingAndOpening(image, kernel_size1,kernel_size2):
    # Create a rectangular kernel
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size2)
    # Apply the closing operation
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1)
    
    # Apply the opening operation
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel2)
        
    return opened_image

def Apply_OpeningAndClosing(image, kernel_size1,kernel_size2):


    
    # Create a rectangular kernel
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size2)
  
    # Apply the opening operation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel2)
    
    # Apply the closing operation
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel1)
    
    return closed_image

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

def Percentage_shades(segment_mask):
    # Calculate the percentage of white and black pixels in the segment mask
    total_pixels = segment_mask.shape[0] * segment_mask.shape[1] 
    white_pixels = np.sum(segment_mask == 255)
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

def Display_and_Save_Images(original_image, segmented_result, image_name):
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Segment result', segmented_result)

    # Save the images to the "results" folder
    result_folder = "./results"
    os.makedirs(result_folder, exist_ok=True)
    original_image_path = os.path.join(result_folder, f"{image_name}_original.jpg")
    segmented_result_path = os.path.join(result_folder, f"{image_name}_segmented.jpg")
    cv2.imwrite(original_image_path, original_image)
    cv2.imwrite(segmented_result_path, segmented_result)
    
    cv2.waitKey(1000)  # Display images for 1 second
    cv2.destroyAllWindows()
    
def get_image_paths_in_folder(folder_path, extensions=[".jpg", ".png", ".jpeg"]):
    image_paths = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    
    return image_paths


if __name__ == "__main__":
    image_paths = get_image_paths_in_folder('./Selection/')
    for elemento in image_paths:
        #Read Image path
        image_name = os.path.splitext(os.path.basename(elemento))[0]
        #Read Image
        original_image = Image_Reading(elemento)
        #Resize Image
        resized_image = cv2.resize(original_image, (800, 600))
        #Filtered by color
        gree_image, green_mask = Apply_Green_Highlight(resized_image)
        #Apply Top-Hat filter
        top_hat_image = Apply_TopHat(gree_image)
        #Green mask application
        green_highlighted_image, green_mask = Apply_Green_Highlight(top_hat_image)
        green_mask = Apply_OpeningAndClosing(green_mask, (15, 15), (10, 10))
        #Segmentation by area
        segment_mask = filter_objects_by_area(green_mask, 500)
        segmented_result = cv2.bitwise_and(resized_image, resized_image, mask=segment_mask)
        #Percentage by area
        Percentage_shades(segment_mask)
        #Results
        Display_and_Save_Images(resized_image, segmented_result, image_name)