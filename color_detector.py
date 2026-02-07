"""
Color Detection Module for Car Color Classification
This module helps detect the dominant color of a car and classify it as blue or not.
"""

import cv2
import numpy as np


def get_dominant_color(image_region):
    """
    Get the dominant color in an image region.
    
    Args:
        image_region: BGR image region (cropped car)
    
    Returns:
        dominant_color: BGR tuple of the dominant color
    """
    # Resize for faster processing
    height, width = image_region.shape[:2]
    if height > 100 or width > 100:
        scale = min(100/height, 100/width)
        image_region = cv2.resize(image_region, None, fx=scale, fy=scale)
    
    # Reshape the image to be a list of pixels
    pixels = image_region.reshape(-1, 3)
    
    # Convert to float
    pixels = np.float32(pixels)
    
    # Define criteria for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = 5  # Number of clusters (we want the top 5 colors for better accuracy)
    
    # Apply k-means
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Get the most common cluster (dominant color)
    counts = np.bincount(labels.flatten())
    dominant_idx = np.argmax(counts)
    dominant_color = centers[dominant_idx]
    
    return tuple(map(int, dominant_color))


def is_blue_color(bgr_color):
    """
    Check if a BGR color is blue (EXTREMELY STRICT to avoid gray/silver/white cars).
    
    Args:
        bgr_color: Tuple of (B, G, R) values
    
    Returns:
        bool: True if the color is blue, False otherwise
    """
    b, g, r = bgr_color
    
    # FIRST: Reject grayscale colors (white, gray, silver, black)
    # If B, G, R are too similar, it's grayscale - NOT blue
    max_diff = max(abs(b-g), abs(b-r), abs(g-r))
    if max_diff < 60:  # If difference is less than 60, it's grayscale
        return False
    
    # Blue must be the HIGHEST channel
    if b <= r or b <= g:
        return False
    
    # Blue must be significantly higher (at least 60 more than others)
    if b < r + 60 or b < g + 60:
        return False
    
    # Blue absolute value must be high (at least 120)
    if b < 120:
        return False
    
    # Red and Green must be relatively low (not white)
    if r > 140 or g > 140:
        return False
    
    # Check the ratio: blue should be at least 1.4x of the max(red, green)
    max_rg = max(r, g)
    if max_rg > 0:
        ratio = b / max_rg
        if ratio < 1.4:
            return False
    
    # HSV validation - VERY STRICT
    color_img = np.uint8([[bgr_color]])
    hsv_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_color
    
    # Hue must be in pure blue range (100-130)
    if not (100 <= h <= 130):
        return False
    
    # Saturation must be HIGH (80+) - reject washed out colors
    if s < 80:
        return False
    
    # Value should be decent (60+) but not too high (avoid white)
    if v < 60 or v > 220:
        return False
    
    # If all checks pass, it's genuinely blue
    return True


def classify_car_color(car_region):
    """
    Classify if a car is blue or not.
    
    Args:
        car_region: BGR image region of the car
    
    Returns:
        str: "blue" if car is blue, "other" otherwise
    """
    try:
        # Get dominant color
        dominant = get_dominant_color(car_region)
        
        # Check if it's blue
        if is_blue_color(dominant):
            return "blue"
        else:
            return "other"
    except Exception as e:
        # If any error occurs, default to "other"
        return "other"
