# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:27:39 2021

This script outputs 6 items in this order:

1. array containing data for each potato in image (convert to java array using Chaquopy)
2. number of potatoes (string)
3. minimum l/w ratio (string)
4. maximum l/w ratio (string)
5. average l/w ratio (string)
6. processed image (encoded as a string, decode using Chaquopy)

The main() function contains three input parameters: The encoded image from 
Chaquopy and a string specifying the coin reference object being used. At this
time the only options are "Quarter", "Dime", or "None". Measurements are in centimeters.
If no reference object is used, the measurements will be returned in pixels.
There is also a good chance it may not work well due to no size thresholding if
no reference object is used.


*This version uses a trained U-Net binary segmentation model for the binary
 thresholding of the image. It has proven to be much more accurate, but slower

            
@author: Alex (alexander.glenn@wsu.edu)
"""

#-----------------------------------------------------------
# These libraries need to be imported in Chaquopy: numpy, opencv, imutils, PIL, skimage, tflite-runtime
import numpy as np
import cv2 
import imutils
import imutils.contours
from PIL import Image
import io
import skimage
import skimage.segmentation
import skimage.morphology
from os.path import dirname, join
import tensorflow as tf # FIXME: tflite does not work on MacOS
#-----------------------------------------------------------

def text_scaling(input_img):
    font_size = int((4*min(input_img.shape[0], input_img.shape[1])/1370))
    font_thickness = int((11*min(input_img.shape[0], input_img.shape[1])/1370))
    line_thickness = int((15*min(input_img.shape[0], input_img.shape[1])/1370))
    return font_size, font_thickness, line_thickness

def coin_type(string):
    
    if string == "Quarter":
        diameter = 2.426 #cm
    elif string == "Dime":
        diameter = 1.791 #cm
    elif string[0] == "C":
        diameter = float(string.split("_")[1]) #cm

    return diameter

def binary_threshold(output_img):
    # turn into grayscale image
    img_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    
    # binary thresholding
    blurred_img = cv2.GaussianBlur(img_gray, (3,3), 0)
    
    # need auto canny values and auto adaptive values:
    v = np.median(blurred_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    block_size = int(min(output_img.shape[0], output_img.shape[1])*0.4)
    if(block_size % 2) == 0:
        block_size = block_size+1
    
    # Find adaptive and normal threshold
    adaptive_thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size,0)
    ret, normal_thresh = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    
    # Find canny threshold
    canny_thresh = cv2.Canny(img_gray, lower, upper)
    
    cnts = cv2.findContours(canny_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = imutils.grab_contours(cnts)
    
    copy = np.zeros_like(output_img)
    
    for c in cnts:
        hull = cv2.convexHull(c) # Find convex hull of canny threshold
        cv2.drawContours(copy, [hull], -1, (255,255,255), -1)
    
    contour_thresh = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    
    # Multiply alll thresholds together
    final_thresh = contour_thresh*normal_thresh*adaptive_thresh
    
    cnts = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = imutils.grab_contours(cnts)
    
    copy2 = np.zeros_like(output_img)
    
    # Find outer contour of final threshold
    for c in cnts:
        hull = cv2.convexHull(c)
        cv2.drawContours(copy2, [hull], -1, (255,255,255), -1)
    
    new_final_thresh = cv2.cvtColor(copy2, cv2.COLOR_BGR2GRAY)
    
    return new_final_thresh

def ref_obj_analysis(coin_string, ref_obj_thresh, markers):
    
    normalized_mask = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    ret, full_potato_mask = cv2.threshold(normalized_mask, 0, 255, cv2.THRESH_BINARY)
    
    x, y, w, h = cv2.boundingRect(full_potato_mask) # bounding area of all potatoes
    
    left = (x, np.argmax(full_potato_mask[:,x])) # only want area left of potatoes
    
    ref_obj_thresh = ref_obj_thresh[:,0:left[0]+100] # splice image
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(ref_obj_thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    opening = skimage.segmentation.clear_border(opening) # ignore objects touching boundaries 
    
    # Watershed needs a sure background, sure foreground, and unknown pixels
    sure_bg = cv2.dilate(opening, kernel, iterations = 2)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255 ,0) # 0.2 = 20%
    
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Watershed needs markers, construct using the sure foreground
    ret2, coin_markers = cv2.connectedComponents(sure_fg)
    
    # By default marker has value of 0, and sure_bg is also 0, so make sure_bg 1
    coin_markers = coin_markers+1
    
    # Create a marker array
    coin_markers[unknown==255] = 0
    
    # Implement watershed
    coin_markers = skimage.segmentation.watershed(-dist_transform, coin_markers)  # FIXME: Was morphology but has to switch to segmentation
    
    normalized_mask = cv2.normalize(coin_markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    ret, full_marker_mask = cv2.threshold(normalized_mask, 0, 255, cv2.THRESH_BINARY)
    
    full_marker_mask = skimage.segmentation.clear_border(full_marker_mask)
    
    cnts = cv2.findContours(full_marker_mask, cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    j = 0
    x_circle = 0
    y_circle = 0
    circle_radius = 0
    coin_area = 0
    pixels_per_cm = 0
    
    if(cnts):
        (cnts, _) = imutils.contours.sort_contours(cnts) # sort from left to right
    
        (x_circle, y_circle), circle_radius = cv2.minEnclosingCircle(cnts[j]) 
        
        coin_diameter = 2*circle_radius
        
        coin_area = cv2.contourArea(cnts[j])
        
        # using the reference object, determine the pixels per cm in the image
        diameter = coin_type(coin_string)
    
        pixels_per_cm = coin_diameter/diameter
    
    return pixels_per_cm, x_circle, y_circle, circle_radius, coin_area



def main(data, coin_string):
    
    np_img = np.asarray(data, np.uint8)
    
    # input_img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
    input_img = np_img  # TODO: Find out why imdecode does not work here
    
    # shrink image if too large (android phones can't process large images well, the newer models can though)
    if(input_img.size > 2800000):
        input_max = np.max(input_img.shape)
        input_scale = 1000/input_max
        input_img = cv2.resize(input_img, None, fx=input_scale,fy=input_scale,interpolation=cv2.INTER_AREA)
        output_img = input_img
    else:
        output_img = input_img
    
    #-----------------------------------------------------------
    """
    preprocessing before implementation of Watershed algorithm
    """
    
    # binary thresholding
    ref_obj_thresh = binary_threshold(output_img)
    
    #-----------------------------------------------------------
    """
    machine learning segmentation
    """
    SIZE_X = 256
    SIZE_Y = 256
    
    origx = output_img.shape[0]
    origy = output_img.shape[1]
    
    test_img = cv2.resize(input_img, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_AREA)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    test_img = np.expand_dims(test_img, axis=0)
    test_img_np = np.array(test_img, dtype=np.float32)
  
    model_name = join(dirname(__file__), "potato_model_4_11_21.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_name)

    interpreter.allocate_tensors() #needs to be before
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.resize_tensor_input(input_details[0]['index'], (1, 256, 256, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (1, 256, 256, 1))
    
    interpreter.set_tensor(input_details[0]['index'], test_img_np)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    new_output_img = np.reshape(output_data, (SIZE_Y, SIZE_X))
    newimg = cv2.resize(new_output_img, (origy, origx), interpolation=cv2.INTER_AREA)
    
    #-----------------------------------------------------------
    """
    watershed
    """
    
    newimg2 = cv2.normalize(newimg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    ret, thresh = cv2.threshold(newimg2, 200, 255, cv2.THRESH_BINARY)
    
    # opening (erosion followed by dilation of binary image)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    opening = skimage.segmentation.clear_border(opening) # ignore objects touching boundaries 
    
    # Watershed needs a sure background, sure foreground, and unknown pixels
    sure_bg = cv2.dilate(opening, kernel, iterations = 2)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255 ,0) # 0.2 = 20%
    
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Watershed needs markers, construct using the sure foreground
    ret2, markers = cv2.connectedComponents(sure_fg)
    
    # By default marker has value of 0, and sure_bg is also 0, so make sure_bg 1
    markers = markers+1
    
    # Create a marker array
    markers[unknown==255] = 0
    
    # Implement watershed
    markers = skimage.segmentation.watershed(-dist_transform, markers)  # FIXME: Was morphology but has to switch to segmentation
  
    #-----------------------------------------------------------
    """
    If a reference object is chosen to be used,
    create a mask of the full marker image and find the reference object
    the reference object should be the left-most object in the image
    """
    
    pixels_per_cm = 0
    
    if(coin_string != "None"):
        pixels_per_cm, x_circle, y_circle, circle_radius, coin_area = ref_obj_analysis(coin_string, ref_obj_thresh, markers)
        
    if(pixels_per_cm == 0):
        coin_string = "None"
        
    #-----------------------------------------------------------
    """
    again use the markers, but this time go through them individually
    to fit bounding rectangles and extract the needed data from the image
    """
    
    font_size, font_thickness, line_thickness = text_scaling(output_img)
    
    potato_arr = []
    ratio_arr = []
    potato_number = 1
    
    for marker in np.unique(markers):
    
        if marker == 1: # 1 means background, ignore it
            continue
        
        marker_mask = np.zeros_like(thresh)
        marker_mask[markers == marker] = 255
        
        cnts = cv2.findContours(marker_mask, 
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if (coin_string != "None"):
            
            if cv2.contourArea(cnts[0]) > coin_area+50: # size thresholding
            
                rectangle = cv2.minAreaRect(cnts[0]) # fit rectangle to contour
                h = (rectangle[1][0])/pixels_per_cm
                w = (rectangle[1][1])/pixels_per_cm
                
                if (w > h):
                    temp = w
                    w = h
                    h = temp
                
                ratio_arr.append(h/w)
                
                box = cv2.boxPoints(rectangle)
                box = np.asarray(box, dtype=np.int32)  # TODO: Why had to change from box = np.int0(box)
                
                x = int(rectangle[0][0])
                y = int(rectangle[0][1])
                
                cv2.drawContours(output_img,[box],-1,(0,0,255),line_thickness)
        
                x_offset = cv2.getTextSize(str(potato_number),
                                           cv2.FONT_HERSHEY_SIMPLEX,font_size,
                                           font_thickness)[0][0]
                
                x_offset = int(x_offset/2)
        
                cv2.putText(output_img,str(potato_number), (x-x_offset,y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,120,0), font_thickness)
                
                potato_arr.append([np.round(h,2),np.round(w,2),np.round(h/w,2)])
                
                potato_number = potato_number+1
                
            output_img = cv2.circle(output_img, (int(x_circle), int(y_circle)), int(circle_radius), (255,0,255), line_thickness)
        
        else:
            
            rectangle = cv2.minAreaRect(cnts[0])
            h = (rectangle[1][0])
            w = (rectangle[1][1])
            
            if (w > h):
                temp = w
                w = h
                h = temp
            
            ratio_arr.append(h/w)
            
            box = cv2.boxPoints(rectangle)
            box = np.asarray(box, dtype=np.int32)
            
            x = int(rectangle[0][0])
            y = int(rectangle[0][1])
            
            cv2.drawContours(output_img,[box],-1,(0,0,255),line_thickness)
        
            x_offset = cv2.getTextSize(str(potato_number),
                                           cv2.FONT_HERSHEY_SIMPLEX,font_size,
                                           font_thickness)[0][0]
                
            x_offset = int(x_offset/2)
        
            cv2.putText(output_img,str(potato_number), (x-x_offset,y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,120,0), font_thickness)            
            
            potato_arr.append([np.round(h,2),np.round(w,2),np.round(h/w,2)])
            
            potato_number = potato_number+1
            
    #-----------------------------------------------------------
    """
    after data is collected, finalize what will be outputted:
        1. array containing data for each potato in image (convert to java array using Chaquopy)
        2. number of potatoes (string)
        3. minimum l/w ratio (string)
        4. maximum l/w ratio (string)
        5. average l/w ratio (string)
        6. processed image (encoded as a string, decode using Chaquopy)
    """
    
    # check if arrays are empty
    if not potato_arr:
        potato_arr = [[0,0,0]]
        
    if not ratio_arr:
        ratio_arr = [0]
    
    ratio_arr = np.array(ratio_arr)
    potato_arr = np.array(potato_arr)
    
    num = float(len(ratio_arr))
    min_ratio = np.round(ratio_arr.min(), 2)
    max_ratio = np.round(ratio_arr.max(), 2)
    avg_ratio = np.round(np.sum(ratio_arr)/num, 2)
    
    num_str = f"{num: .2f}"
    min_str = f"{min_ratio: .2f}"
    max_str = f"{max_ratio: .2f}"
    avg_str = f"{avg_ratio: .2f}"
    
    fixed_output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(fixed_output_img)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    
    
    #return potato_arr, "%.2f"%num, "%.2f"%min_ratio, "%.2f"%max_ratio, "%.2f"%avg_ratio, buff.getvalue()
    #return potato_arr, ""+str(num), ""+str(min_ratio), ""+str(max_ratio), ""+str(avg_ratio), buff.getvalue()
    return  potato_arr, num_str+"", min_str+"", max_str+"", avg_str+"", buff.getvalue()

# ___________________________________ TEMP ___________________________________
import cv2
import numpy as np
import os

def run_on_test_image():
    # Path to your image file
    image_path = "/Users/trevorbuchanan/PycharmProjects/PotatoApp/MachineLearning/test_image.jpg"  # Ensure this path is correct
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return
    
    # Read the image using OpenCV
    input_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Directly use OpenCV to load the image
    
    if input_img is None:
        print(f"Error: Failed to load image from {image_path}")
        return
    
    # Print the shape of the loaded image to verify it was loaded correctly
    print(f"Image shape: {input_img.shape}")
    
    # Call the main function with the coin reference type (e.g., "Quarter")
    coin_string = "Quarter"  # Change this to the desired coin type (e.g., "Dime" or "C_2.5")
    potato_arr, num_str, min_str, max_str, avg_str, buff_img = main(input_img, coin_string)
    
    print(f"Potato arr: {potato_arr}")
    print(f"Num str: ({num_str})")
    print(f"Min str: {min_str}")
    print(f"Max str: {max_str}")
    print(f"Avg str: {avg_str}")
    
    # Convert the byte buffer to an image
    image_array = np.frombuffer(buff_img, dtype=np.uint8)
    result_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if result_img is None:
        print("Error: Failed to decode the image buffer.")
        return
    
    # Example: Save the processed image with the potatoes outlined
    output_image_path = "/Users/trevorbuchanan/PycharmProjects/PotatoApp/MachineLearning/result_image.jpg"
    
    # Save the result image to a file
    cv2.imwrite(output_image_path, result_img)
    print(f"Result image saved to {output_image_path}")

# Run the function to process the image
run_on_test_image()