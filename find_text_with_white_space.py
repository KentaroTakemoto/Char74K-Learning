# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 18:44:59 2016

@author: 武本
"""

#coding:utf-8
import numpy as np
import time as time
import sys
import pylab as pl
import pandas as pd
#from mlp import MultiLayerPerceptron
#from sklearn.datasets import fetch_mldata
#from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import LabelBinarizer
#from sklearn.metrics import confusion_matrix, classification_report
#from matplotlib import pyplot as plt
#from matplotlib import cm
from scipy.misc import imread,imresize
#import csv
import cv2
from sklearn.externals import joblib

learning_model_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\Learning_Models\\K-Neighbor(changed1)_resized_new(20)'

clf = joblib.load(learning_model_path)


# image_path ='C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\pictures\\mediaseek.png'
image_path ='C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\pictures\\marks\\AozoraMinchobold.png'

#image_path ='C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\Char74K\\EnglishFnt\\English\\Fnt\\Sample006\\img006-01016.png'
########################### obtain image ###########################
img = cv2.imread(image_path)
if len(img.shape) == 3:
    img_height, img_width, img_channels = img.shape[:3]
else:
    img_height, img_width = img.shape[:2]
    img_channels = 1

########################### convert image ###########################
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# print(thresh)

########################### 上下の幅を調節 ###########################
# 横に一列すべて255となる白線を検出
# 左が白線なのに白線でなくなる座標、左が白線でないのに白線になる座標をそれぞれ(list)scan_start, (list)scan_endに記録する
scanx_start = []
scanx_end = []
max_of_bright = img_width*255
sum_of_bright = []
sum_of_bright.append(max_of_bright)

for scan in range(1,img_height):
    a = 0
    for scan_x in range(0,img_width):
        a += thresh[scan, scan_x]
    sum_of_bright.append(a)

    if sum_of_bright[scan] <= (max_of_bright-255) and sum_of_bright[scan-1] > (max_of_bright-255):
        scanx_start.append(scan)
    elif sum_of_bright[scan] > (max_of_bright-255) and sum_of_bright[scan-1] <= (max_of_bright-255):
        scanx_end.append(scan)
if scanx_end == []:
    scanx_end.append(img_height)

thresh_resize = thresh[max(0, (scanx_start[0]-(int)(scanx_end[-1]-scanx_start[0])/20)) : min(img_height-1, scanx_end[-1]+(int)(scanx_end[-1]-scanx_start[0])/20) , : ]  


if len(thresh_resize.shape) == 3:
    thresh_resize_height, thresh_resize_width, thresh_resize_channels = thresh_resize.shape[:3]
else:
    thresh_resize_height, thresh_resize_width = thresh_resize.shape[:2]
    thresh_resize_channels = 1

########################### scanning ###########################
# 縦に一列、すべての画素値が255(=白線)になるところを検出する
# 左が白線なのに白線でなくなる座標、左が白線でないのに白線になる座標をそれぞれ(list)scan_start, (list)scan_endに記録する
scan_start = []
scan_end =[]

max_of_bright = thresh_resize_height*255
sum_of_bright = []
sum_of_bright.append(max_of_bright)

for scan in range(1,thresh_resize_width):
    a = 0
    for scan_y in range(0,thresh_resize_height):
        a += thresh_resize[scan_y, scan]
    sum_of_bright.append(a)

    if sum_of_bright[scan] <= (max_of_bright-255) and sum_of_bright[scan-1] > (max_of_bright-255):
        scan_start.append(scan)
    elif sum_of_bright[scan] > (max_of_bright-255) and sum_of_bright[scan-1] <= (max_of_bright-255):
        scan_end.append(scan)
               
#print (scan_start) 
#print (scan_end)

########################### detected ###########################
detected_images = []
predict_string = []     #予測された文字列（URL)

for detect in range(0,len(scan_start)):
    
    # scan_start[detect]とscan_end[detect]の間の領域が、detect番目に発見された文字領域である
    detected_image = thresh_resize[0:thresh_resize_height,scan_start[detect]:scan_end[detect]]  
    if len(detected_image.shape) == 3:
        detect_height, detect_width, detect_channels = detected_image.shape[:3]
    else:
        detect_height, detect_width = detected_image.shape[:2]
        detect_channels = 1    
    #pl.matshow(detected_images[detect])

    detected_images.append(detected_image)
    
    if detect_width <= detect_height : 
        resize = np.ones([detect_height,detect_height])
        resize = resize*255
        
        resize[0:detect_height , (detect_height/2-detect_width/2):(detect_height/2+detect_width/2)] = detected_image
       
        detected_resize = imresize(resize, (20,20) )
    else:
        resize = np.ones([detect_width,detect_width])
        resize = resize*255
        
        resize[(detect_width/2-detect_height/2):(detect_width/2+detect_height/2) , 0:detect_width] = detected_image
       
        detected_resize = imresize(resize, (20,20) )
        
        
        
#        resize = imresize(detected_image,(20.0/detect_width))
#        if len(resize.shape) == 3:
#            resize_height, resize_width, resize_channels = resize.shape[:3]
#        else:
#            resize_height, resize_width = resize.shape[:2]
#            resize_channels = 1        
#       
#        detected_resize =  np.ones([20,20],dtype=np.uint8 )
#        detected_resize = detected_resize*255
#        detected_resize[((20/2)-resize_height/2):((20/2)+resize_height/2),0:20] = resize
    
    pl.matshow(detected_resize)
    a = np.reshape(detected_resize,(detected_resize.shape[0]*detected_resize.shape[1]))           
    predict = str(clf.predict(a)[0])
    
    predict_string.append(predict)
    
print(predict_string)



#
##ret,thresh = cv2.threshold(gray,127,255,0)
#im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
##################      Now finding Contours         ###################
#
##contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
##contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##contours = cv2.findContours(thresh,mode = cv2.RETR_TREE,method = cv2.CHAIN_APPROX_SIMPLE)
#
#samples =  np.empty((0,100))
#responses = []
#keys = [i for i in range(48,58)]




#
#result=pd.DataFrame(img_height*[img_width*[' ']])
#
#for cnt in contours:
#    if cv2.contourArea(cnt)>50:
#        [x,y,w,h] = cv2.boundingRect(cnt)
#
#        if  (h>20 and w>20) and (h<img_height-5 and w<img_width-5):
#            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
#            roi = thresh[y:y+h,x:x+w]
#            roismall = cv2.resize(roi,(10,10))  
#            
#            x = x-3
#            y = y-3
#            w = w+6
#            h = h+6
#            
#            detected_image = gray[y:y+h,x:x+w]
#            
#            detected_resize = imresize(detected_image,[20,20])
#            
#            a = np.reshape(detected_resize,(detected_resize.shape[0]*detected_resize.shape[1]))
#            
#            predict = str(clf.predict(a)[0])
#            
#            for i in range(0,w):
#                for j in range(0,h):
#                    result.set_value((y+j),(x+i),predict)
#            
#            cv2.imshow('norm', img)
#            key = cv2.waitKey(2)
#            pl.matshow(detected_resize)
#
#csv_result_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\results\\find_letter_in_pic\\alphabet21.csv'
#
##ResultMatrix=pd.DataFrame(result)
#result.to_csv(csv_result_path,index=0,header=None)
# 