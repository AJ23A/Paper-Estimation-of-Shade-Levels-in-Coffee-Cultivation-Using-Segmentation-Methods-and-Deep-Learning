import cv2
import numpy as np

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
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    
    # Create a mask for green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Create a black image of the same size as input
    highlight_image = np.zeros_like(image)
    
    # Set the green region in the highlight image
    highlight_image[green_mask > 0] = [0, 255, 0]  # Highlight in green
    
    # Combine the highlight image and the original image
    highlighted_image = cv2.addWeighted(image, 0.7, highlight_image, 0.3, 0)
    
    return highlighted_image

if __name__ == "__main__":
    image_path = './Selection/Agricam_01F.JPG'

    original_image = Image_Reading(image_path)
    resized_image = cv2.resize(original_image, (800, 600))

    green_highlighted_image = Apply_Green_Highlight(resized_image)

    # Display the images
    cv2.imshow('Original Image', resized_image)
    cv2.imshow('Green Highlighted Image', green_highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
