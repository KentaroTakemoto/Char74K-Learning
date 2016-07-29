# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:27:16 2016

@author: Nantan
"""

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def fisher_discriminant_analysis(image):
    _image = image.reshape(image.shape[0] * image.shape[1])
    ave = np.average(_image)
    j = 0 #評価関数初期値
    
    #閾値探索
    for _t in range(np.min(_image) -1, np.max(_image) + 1):
        blacks = _image[_image < _t]
        whites = _image[_image >= _t]
        inter_var = len(blacks) * np.power(np.average(blacks) - ave, 2)/len(_image) + len(whites) * np.power(np.average(whites) - ave, 2)/len(_image)
        _j = inter_var/(np.var(_image)-inter_var)
        #閾値の更新
        if _j > j:
            t = _t
            j = _j
    
    #画像を二値化
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] >= t:
                image[i][j] = 255
            else:
                image[i][j] = 0
                
    return image #二値化後の画像配列を返す
    
if __name__ == '__main__':
    image = fisher_discriminant_analysis(np.array(Image.open("sample.jpg").convert('L'), dtype = np.uint8))
    plt.imshow(image, cmap = 'Greys_r')
    plt.show()
    image.save('sample_after.jpg')