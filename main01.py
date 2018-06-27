import cv2
import numpy as np
import os
import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
import PIL

''' global variable 'img.jpg' '''
global img_test

def ReadTestImg():
    global img_test
    img_test = cv2.imread('img.jpg')

def Gaussfun(x,y,sigma=1):
    '''Gaussfun function'''
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))
    # a**b = a^b


def main():
    #get test img
    global img_test
    ReadTestImg()

    '''get test img shape(including the number of rows of pix and such on)'''
    '''t dimensions of imgtest = 648*401'''
    (row_test, col_test, cal_test) = img_test.shape

    (aim_row,aim_col) = (2*row_test,2*col_test)
    newsizeImg = cv2.resize(img_test,(aim_col,aim_row))
    #cv2.imshow('test',newsizeImg)
    #cv2.waitKey(0)
    #cv2.destroyWindow('test')

    #生成标准差为5的5*5高斯算子
    suanzi_Gauss = np.fromfunction(Gaussfun,(6,6),sigma = 6)
    #生成拉普拉斯拓展算子
    suanzi_Laplace = np.array([[1,1,1],[1,-8,1],[1,1,1]])

    img_Gray = cv2.cvtColor(newsizeImg,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('test',img_Gray)
    #cv2.waitKey(0)
    #cv2.destroyWindow('test')
    img_array = np.array(img_Gray)

    #卷积运算（高斯算子)
    img_blur = signal.convolve2d(img_array,suanzi_Gauss,mode="same")

    #卷积运行(拉普拉斯算子)
    img_blur_2 = signal.convolve2d(img_blur,suanzi_Laplace,mode="same")
    
    #结果转化为0-255
    img_blur_2 = (img_blur_2/float(img_blur_2.max()))*255
    print(img_blur_2)
    img_blur_2[img_blur_2>img_blur_2.mean()-12] = 255
    img_blur_2[img_blur_2<img_blur_2.mean()-450] = 0

    plt.subplot(2,1,1)
    plt.imshow(img_Gray,cmap=cm.gray)
    plt.axis("off")
    plt.subplot(2,1,2)
    plt.imshow(img_blur_2,cmap=cm.gray)
    plt.axis("off")
    plt.show()

    
if __name__ == '__main__':
    main()   
