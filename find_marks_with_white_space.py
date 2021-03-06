# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:45:10 2016

@author: 武本
"""

#coding:utf-8
import numpy as np
import time as time
import sys
import pylab as pl
import pandas as pd
from scipy.misc import imread,imresize
import cv2
import random

#csvに書き込むための画素値データを格納する配列
#datas[0]にhyphen, datas[1]にunderbar・・・datas[18]にpercent というふうにデータを格納する
#この書き方アホっぽいからなんとかしたい
datas = [np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),
         np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),
        np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),
        np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),
        np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),
        np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),np.empty((0,(20*20)), str),
        np.empty((0,(20*20)), str)]

#今回用いたフォントの名前 10種類
#http://lightbox.on.coocan.jp/  からお借りしました
#なるべくURLに使われてそうなフォントを選んだ
font_names = ['AozoraMinchobold','tkaisho-gt01','ume-pmo3','roundedmplus2clight','kokugl',
              'dasaji_win','NagomiGokubosoGothic','jiyunotsubasa','jkg-l_2','wlmaru2004']

image_path ='C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\pictures\\marks\\'
#image_path ='C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\Char74K\\EnglishFnt\\English\\Fnt\\Sample006\\img006-01016.png'

for i in range(0,10):
    image_path_ = image_path + str(font_names[i]) + '.png'

    ########################### obtain image ###########################
    img = cv2.imread(image_path_)
    if len(img.shape) == 3:
        img_height, img_width, img_channels = img.shape[:3]
    else:
        img_height, img_width = img.shape[:2]
        img_channels = 1
    
    ########################### 画像を2値化 ###########################
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
    
        #以下の判定にmax_of_brightをそのまま用いてもよかったのだが、ノイズが入ることを考えて少しゆとりをもたせた
        # noiseはノイズ係数　ノイズが多そうなときは大きくするといいかも　ただし目安として画像サイズの 1/10以下程度にしないと機能しない
        noise = 1
        if sum_of_bright[scan] < (max_of_bright-255*noise) and sum_of_bright[scan-1] >= (max_of_bright-255*noise):
            scanx_start.append(scan)
        elif sum_of_bright[scan] >= (max_of_bright-255*noise) and sum_of_bright[scan-1] < (max_of_bright-255*noise):
            scanx_end.append(scan)
    
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
    
        #以下の判定にmax_of_brightをそのまま用いてもよかったのだが、ノイズが入ることを考えて少しゆとりをもたせた
        # noiseはノイズ係数　ノイズが多そうなときは大きくするといいかも　ただし目安として画像サイズの 1/10以下程度にしないと機能しない
        noise = 1
        if sum_of_bright[scan] < (max_of_bright-255*noise) and sum_of_bright[scan-1] >= (max_of_bright-255*noise):
            scan_start.append(scan)
        elif sum_of_bright[scan] >= (max_of_bright-255*noise) and sum_of_bright[scan-1] < (max_of_bright-255*noise):
            scan_end.append(scan)
                   
    #print (scan_start) 
    #print (scan_end)
    
    ########################### detected ###########################
    detected_images = []
    
    for detect in range(0,len(scan_start)):
        
        # scan_start[detect]とscan_end[detect]の間の領域が、detect番目に発見された文字領域である
        detected_image = thresh_resize[0:thresh_resize_height,scan_start[detect]:scan_end[detect]]  
        if len(detected_image.shape) == 3:
            detect_height, detect_width, detect_channels = detected_image.shape[:3]
        else:
            detect_height, detect_width = detected_image.shape[:2]
            detect_channels = 1    
        #pl.matshow(detected_images[detect])  
  
          
        # detected_imagesの0番目はhyphen,1番目はunderbarを含む画像領域、といったふうになる    
        detected_images.append(detected_image)
        
        ########################### resize ###########################
        # 得られた領域を長辺サイズの正方領域内に配置し、最後にimresize
        if detect_width <= detect_height : 
            
            #正方領域の初期化
            resize = np.ones([detect_height,detect_height],dtype=np.uint8)
            resize = resize*255
            
            #正方領域内に配置
            resize[0:detect_height , (detect_height/2-detect_width/2):(detect_height/2+detect_width/2)] = detected_image
            
            #imresize
            detected_resize = imresize(resize, (20,20) )
            
        else:   #detect_width > detect_height
            resize = np.ones([detect_width,detect_width],dtype=np.uint8)
            resize = resize*255
            
            resize[(detect_width/2-detect_height/2):(detect_width/2+detect_height/2) , 0:detect_width] = detected_image
           
            detected_resize = imresize(resize, (20,20) )
            
            
        #pl.matshow(detected_resize)
        
        ########################### multiply data ###########################
        #縦横それぞれ±2ピクセルの範囲に画像をずらすー＞25倍 
        #全ての画素値をランダムに±10するというパターンを4種類生成(カメラ認識を見据えて、ノイズを加える)ー＞4倍
        #合わせて100倍にデータを増やす
        
        for dx in range(-2,3):
            for dy in range(-2,3):
                for noise in range(0,4):
                    random.seed()
                    
                    ########画像を縦横にずらす##########
                    #ずらした領域の初期化
                    detected_shift = np.ones([20,20],dtype=np.uint8)
                    detected_shift = detected_shift*255                     
                    
                    #元画像をずらして配置
                    detected_shift[max(0,dx):min(20,20+dx) , max(0,dy):min(20,20+dy)] = detected_resize[max(0,-dx):min(20,20-dx) , max(0,-dy):min(20,20-dy)]
                    
                    ########ノイズを加える##########
                    for x in range(0,20):
                        for y in range(0,20):
                            #0~255の範囲から出ないように画素値を±10する
                            b = min(　max(　0,detected_shift[x][y] + random.randint(-10,10)　),　255　)
                            detected_shift[x][y] = int(b)
                    
                    #pl.matshow(detected_shift)
                    
                    ########画像データを記録する##########
                    a = np.reshape(detected_shift,(detected_shift.shape[0]*detected_shift.shape[1]))           
                    a_ = [a]
                    datas[detect] = np.append(datas[detect], a_, axis=0)
            
        #print('finish')

########################### write to CSV ###########################
csv_result_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\results\\data_for_marks\\'

#今回用いた文字列の一覧
#サンプル画像はこの順番に記号が並んでいる
name_of_marks = ['hyphen','underbar','dot','exclamation','asterisk',' apostrophe','op_parenthesis',
                 'cl_parenthesis','semi-colon','slash','question','colon','at','and','equal','plus',
                 'dollar','comma','percent']

for i in range(0,19):
    csv_result_path_ = csv_result_path + str(name_of_marks[i]) + '(20).csv'
    ResultMatrix=pd.DataFrame(datas[i])
    ResultMatrix.to_csv(csv_result_path_,index=0,header=None)
         