# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:03:07 2022

@author: Mohamed ELsheikh
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

# my_image = "lenna.png"
# img=cv2.imread(my_image)

# method Read the image 
def Read_image (img):
    image = cv2.imread(img)
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # plt.show()
    return image
#Read_image(my_image)

#  method convert the image to RGB image 
def convertimage_toRGB (img):
    new_image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10,10))
    # plt.imshow(new_image)
    # plt.show()
    return new_image
#convertimage_toRGB(img)

#  method convert the image to gray scale image
def convertimage_tograyscale (img):
    image_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_gray, cmap='gray')
    # plt.show()
    return image_gray
#convertimage_tograyscale(img)

#  method flip the image vertically
def flipimage_vertically (img):
    img_flip_ud = cv2.flip(img, 0)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_flip_ud)
    # plt.show()
    return img_flip_ud
#flipimage_vertically(img)

#  method flip the image Horizontally
def flipimage_Horizontally(img):
    img_flip_lr = cv2.flip(img, 1)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_flip_lr)
    # plt.show()
    return img_flip_lr 
#flipimage_Horizontally(img)   

 # method crop the image vertically
def cropimage_vertically(img):
    upper = 150
    lower = 400
    crop_top = img[upper: lower,:,:]
    # plt.figure(figsize=(10,10))
    # plt.imshow(cv2.cvtColor(crop_top, cv2.COLOR_BGR2RGB))
    # plt.show()
    return crop_top
#cropimage_vertically(img)

 #  method crop the image horizontally
def cropimage_horizontally(img):
    left = 150
    right = 400
    crop_horizontal =img[: ,left:right,:]
    # plt.figure(figsize=(5,5))
    # plt.imshow(cv2.cvtColor(crop_horizontal, cv2.COLOR_BGR2RGB))
    # plt.show()
    return crop_horizontal 
#cropimage_horizontally(img)

 #  method plot the histogram for image
def plot_histogram(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    # plt.plot(hist)
    # plt.show()
    return hist
#plot_histogram(img) 

#  method adjust Brightness and contrast for image
def brightness_adj(img):
    alpha = 1 # Simple contrast control
    beta = 100   # Simple brightness control   
    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    return  new_image

def contrast_adj(img):
    alpha = 2 # Simple contrast control
    beta = 1   # Simple brightness control   
    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    return  new_image
#brightness_contrast_adj(img)

#  method apply adaptive threshold and histogram equalization on the image

def histogram_equ(img):
    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(src)
    # cv2.imshow('Source image', src)
    # cv2.imshow('Equalized Image', dst)
    # cv2.waitKey(0)
    return dst
#histogram_equ(img)
                          
# method apply scaling, translation and rotation on the image
def scaling(img):
    new_image = cv2.resize(img,None,fx=2, fy=1, interpolation = cv2.INTER_NEAREST )
    # plt.imshow(new_image,cmap='gray')
    # plt.show()
    return new_image 

#scaling(img)

def translation(img):
    tx = 50
    ty = 50
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    rows, cols, _ = img.shape
    new_image = cv2.warpAffine(img, M, (cols+tx , rows +ty))
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    return new_image

#translation(img)

def rotation(img):
    theta = 45.0
    cols, rows, _ = img.shape
    M = cv2.getRotationMatrix2D(center=(cols // 2 - 1, rows // 2 - 1), angle=theta, scale=1)
    new_image = cv2.warpAffine(img, M, (cols, rows))
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    return new_image

#rotation(img)
 
# methods smooth the image using low pass filter 
def low_passfilter (img):
    new_image = cv2.blur(img,(5,5),0)
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    return new_image
    
#low_passfilter(img)

# methods apply gaussian filter and average filter on the image
def gaussian_filter(img):
    new_image = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    return new_image

#gaussian_filter(img)

def average_filter(img):
    new_image= cv2.blur(img,(5,5))
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    return new_image

# average_filter(img)


#methods apply sobel edge detector
def sobel_edgedetector (img):
    image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_image_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    new_image_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_new_image_x = cv2.convertScaleAbs(new_image_x)
    abs_new_image_y = cv2.convertScaleAbs(new_image_y)
    new_image = cv2.addWeighted(abs_new_image_x, 0.5, abs_new_image_y, 0.5, 0)
    #   plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    #   plt.show()
    return new_image
    
#sobel_edgedetector(img) 
    
#methods apply median filter
def median_filter(img):
    new_image = cv2.medianBlur(img,5)
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    return new_image

#median_filter(img)

# methods apply erosion and dilation on the image 

def erosion (img):
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    #    plt.imshow(cv2.cvtColor(img_erosion, cv2.COLOR_BGR2RGB))
    #    plt.show()
    return img_erosion

#erosion(img)

def dilation (img):
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    #     plt.imshow(cv2.cvtColor( img_dilation, cv2.COLOR_BGR2RGB))
    #     plt.show()
    return img_dilation
    
#dilation(img)    







 