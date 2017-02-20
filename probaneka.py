import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from skimage.io import imread
import time
from scipy import ndimage
import math
from skimage.color import rgb2gray
from sklearn.datasets import fetch_mldata
from skimage.morphology import square, diamond, disk
from skimage.morphology import dilation
from skimage.measure import label
from skimage.measure import regionprops
import cPickle, gzip, numpy
from scipy import ndimage
import math
from sklearn.datasets import fetch_mldata

#distanca
def distance(x,y):
    return np.sum(np.logical_xor(x,y))

#transformacija slike radi lakse prepoznavanja
def transformacija(sl):                        
    northern = north(sl)
    western = west(sl)
    dimensions = (28,28)
    nova_slika = np.zeros(dimensions)
    nova_slika[0:28 - northern, 0:28 - western] = sl[northern:28, western:28]
    return nova_slika

def north(sl):
    dim = sl.shape
    for r in range(0,dim[0]):
        for c in range(0, dim[1]):
            if(sl[r,c]==1):
                return r
    return 0

def south(sl):
    dim = sl.shape
    for r in range(0,dim[0]):
        for c in range(0, dim[1]):
            if(sl[dim[0]-1-r,c]==1):
                return dim[0]-1-r
    return 0

def west(sl):
    dim = sl.shape
    for c in range(0,dim[1]):
        for r in range(0, dim[0]):
            if(sl[r,c]==1):
                return c
    return 0

def east(sl):
    dim = sl.shape
    for c in range(0,dim[1]):
        for r in range(0, dim[0]):
            if(sl[r,dim[1]-1-c]==1):
                return dim[1]-1-c
    return 0




#opis regiona
class Regioni:
    bbox = []
    code = 0
    intersected = False
    image = []
    
    def __init__(self, bbox, intersected, code, image):
        self.bbox = bbox
        self.intersected = intersected
        self.code = code
        self.image = image
        
    def update(self, centroid, bbox, rect):
        self.bbox = bbox



#lista regiona
def dodaj_region(objekat):
    global brojac_reg
    global svi_regioni
    
    for i in xrange(len(svi_regioni)):
        trenutni_region = svi_regioni[i].bbox
        w_raz = abs(trenutni_region[0] - objekat.bbox[0])
        h_raz = abs(trenutni_region[1] - objekat.bbox[1])
        raz = w_raz + h_raz
        
        if raz < 30:
            objekat.code = svi_regioni[i].code
            objekat.image = svi_regioni[i].image
            
            if objekat.intersected == False and svi_regioni[i].intersected == True:
                objekat.intersected = True
            svi_regioni[i] = objekat
            return           
    objekat.code = brojac_reg
    brojac_reg = brojac_reg + 1
    svi_regioni.append(objekat)


svi_regioni = []
brojac_reg = 0

#set podataka
fajl = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(fajl)
fajl.close()
mnist_image = train_set[0]
mnist_number = train_set[1]

#prilagodjavnje seta
for brojac in range(0, 50000):
    skl = mnist_image[brojac]
    skl = skl.reshape(28, 28)
    skl = skl < 0.5
    skl = 1 - skl
    skl = transformacija(skl)
    mnist_image[brojac] = skl.reshape(28*28)
  
#video
#cap = cv2.VideoCapture("train_video\\video-5.avi")  


#fajl
text_file = open("out.txt", "w")

#prolazak kroz svaki video
for i in range(0, 10):
    svi_regioni = []
    vid = 'video-' + `i` + '.avi'
    cap = cv2.VideoCapture(vid) 
    #svi_regioni = []
    #prolazak kroz svaki frejm
    for j in range(0, int(cap.get(7))):
        ret, frame = cap.read()
        
        #svaki 30ti frejm
        if ret is True and j % 10 == 0:
            
            #u prvom frejmu pronadji liniju (uvek na istom mestu, uvek iste boje)
            if j == 0:
                img = frame
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(img_gray, 10, 255, 0)
                thresh = 20
                img_b = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
                edges = cv2.Canny(img, 50, 150, apertureSize = 3)
                minLineLength = 100
                maxLineGap = 10
                lines = cv2.HoughLinesP(img_b, 1, np.pi/180, 100, minLineLength, maxLineGap)
                blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                for x1, y1, x2, y2 in lines[0]:
                    cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                cv2.imwrite('houghline.jpg', blank_image)
            
            
            hough = imread('houghline.jpg')
            hough_gray = rgb2gray(hough)
            hough_b = hough_gray < 0.01
            hough_b = 1 - hough_b
            
            img = frame
            img_g = rgb2gray(img)
            img_gg =img_g < 0.75
            img_gg = 1 - img_gg
            
          
            str_obj = disk(3)
            sl_dil = dilation(img_gg, selem = str_obj)
            
            limg = label(sl_dil)
            regions = regionprops(limg)
            
            for region in regions:
                b = region.bbox
                h = b[2] - b[0]
                w = b[3] - b[1]
                
                if w > 13 or h > 13:
                    picture = hough[b[0]:b[2], b[1]:b[3]]
                    picture2 = img_gg[b[0]:b[2], b[1]:b[3]]
                    maska = np.zeros((28,28))
                    s_dim = picture2.shape
                    s_width = min(28,s_dim[0])
                    s_height = min(28, s_dim[1])
                    maska[0:s_width,0:s_height] = picture2[0:s_width,0:s_height]
                    maska = transformacija(maska)
                    prolaz = len(np.unique(picture)) > 5
                    new_obj = Regioni(b,prolaz,0,maska)
                    dodaj_region(new_obj)
                    #print new_elem
                    
        zbir = 0
        
        
    for obj in svi_regioni:
        if obj.intersected:
            minimalna = 1000
            minindex = 0
            tindex = 0
            for trenutna in mnist_image:
                dist = distance(obj.image, trenutna.reshape(28, 28))
                if dist < minimalna:
                    minimalna = dist
                    minindex = tindex
                tindex = tindex + 1
            zbir = zbir + mnist_number[minindex]
            print mnist_number[minindex]
    print zbir            
    text_file.write('video-' + `i` + '.avi\t' + `zbir` + "\n")


    #print zbir
    
    cap.release()
    
    
    
    
    

    
    
    
    
    
    
    
    