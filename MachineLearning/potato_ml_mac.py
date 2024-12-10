# -*- coding: utf-8 -*-
import numpy as np
import cv2
import imutils
import imutils.contours
from PIL import Image
import io
import skimage
import skimage.segmentation
import skimage.morphology
import tensorflow as tf  # Use TensorFlow instead of tflite-runtime 
from os.path import dirname, join

# Text scaling function
def text_scaling(input_img):
    font_size = int((4*min(input_img.shape[0], input_img.shape[1])/1370))
    font_thickness = int((11*min(input_img.shape[0], input_img.shape[1])/1370))
    line_thickness = int((15*min(input_img.shape[0], input_img.shape[1])/1370))
    return font_size, font_thickness, line_thickness

# Coin type reference
def coin_type(string):
    if string == "Quarter":
        diameter = 2.426 # cm
    elif string == "Dime":
        diameter = 1.791 # cm
    elif string[0] == "C":
        diameter = float(string.split("_")[1]) # cm
    return diameter

# Binary thresholding for segmentation
def binary_threshold(output_img):
    img_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(img_gray, (3, 3), 0)
    v = np.median(blurred_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    block_size = int(min(output_img.shape[0], output_img.shape[1]) * 0.4)
    if(block_size % 2) == 0:
        block_size += 1
    
    adaptive_thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 0)
    ret, normal_thresh = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny_thresh = cv2.Canny(img_gray, lower, upper)
    
    cnts = cv2.findContours(canny_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    copy = np.zeros_like(output_img)
    
    for c in cnts:
        hull = cv2.convexHull(c)
        cv2.drawContours(copy, [hull], -1, (255, 255, 255), -1)
    
    contour_thresh = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    final_thresh = contour_thresh * normal_thresh * adaptive_thresh
    
    cnts = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    copy2 = np.zeros_like(output_img)
    
    for c in cnts:
        hull = cv2.convexHull(c)
        cv2.drawContours(copy2, [hull], -1, (255, 255, 255), -1)
    
    new_final_thresh = cv2.cvtColor(copy2, cv2.COLOR_BGR2GRAY)
    return new_final_thresh

# Analysis with reference object
def ref_obj_analysis(coin_string, ref_obj_thresh, markers):
    normalized_mask = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    ret, full_potato_mask = cv2.threshold(normalized_mask, 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(full_potato_mask)
    left = (x, np.argmax(full_potato_mask[:, x]))
    ref_obj_thresh = ref_obj_thresh[:, 0:left[0] + 100]
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(ref_obj_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opening = skimage.segmentation.clear_border(opening)
    
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    ret2, coin_markers = cv2.connectedComponents(sure_fg)
    coin_markers = coin_markers + 1
    coin_markers[unknown == 255] = 0
    coin_markers = skimage.morphology.watershed(-dist_transform, coin_markers)
    normalized_mask = cv2.normalize(coin_markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    ret, full_marker_mask = cv2.threshold(normalized_mask, 0, 255, cv2.THRESH_BINARY)
    full_marker_mask = skimage.segmentation.clear_border(full_marker_mask)
    
    cnts = cv2.findContours(full_marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    j = 0
    x_circle = 0
    y_circle = 0
    circle_radius = 0
    coin_area = 0
    pixels_per_cm = 0
    
    if(cnts):
        (cnts, _) = imutils.contours.sort_contours(cnts)
        (x_circle, y_circle), circle_radius = cv2.minEnclosingCircle(cnts[j])
        coin_diameter = 2 * circle_radius
        coin_area = cv2.contourArea(cnts[j])
        diameter = coin_type(coin_string)
        pixels_per_cm = coin_diameter / diameter
    
    return pixels_per_cm, x_circle, y_circle, circle_radius, coin_area

# Main function
def main(data, coin_string):
    np_img = np.asarray(data, np.uint8)
    input_img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

    if input_img.size > 2800000:
        input_max = np.max(input_img.shape)
        input_scale = 1000 / input_max
        input_img = cv2.resize(input_img, None, fx=input_scale, fy=input_scale, interpolation=cv2.INTER_AREA)
        output_img = input_img
    else:
        output_img = input_img
    
    ref_obj_thresh = binary_threshold(output_img)

    # Machine learning segmentation
    SIZE_X = 256
    SIZE_Y = 256
    origx = output_img.shape[0]
    origy = output_img.shape[1]
    
    test_img = cv2.resize(input_img, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_AREA)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    test_img = np.expand_dims(test_img, axis=0)
    test_img_np = np.array(test_img, dtype=np.float32)
  
    # Load your model from file
    model_name = "potato_model_4_11_21.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.resize_tensor_input(input_details[0]['index'], (1, 256, 256, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (1, 256, 256, 1))
    interpreter.set_tensor(input_details[0]['index'], test_img_np)
    
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    new_output_img = np.reshape(output_data, (SIZE_Y, SIZE_X))
    newimg = cv2.resize(new_output_img, (origy, origx), interpolation=cv2.INTER_AREA)
    
    newimg2 = cv2.normalize(newimg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    ret, thresh = cv2.threshold(newimg2, 200, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opening = skimage.segmentation.clear_border(opening)
    
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    ret2, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = skimage.morphology.watershed(-dist_transform, markers)
  
    pixels_per_cm = 0
    if(coin_string != "None"):
        pixels_per_cm, x_circle, y_circle, circle_radius, coin_area = ref_obj_analysis(coin_string, ref_obj_thresh, markers)
  
    return pixels_per_cm, x_circle, y_circle, circle_radius, coin_area
