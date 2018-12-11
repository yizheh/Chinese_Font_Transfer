import numpy as np
import matplotlib as plt
from PIL import Image
import os
import sys

element_num = 64*64*3

def evaluate(path='/Users/hangyizhe/GitHub/Chinese_Font_Transfer/models/Results/V1.2/CE_32b_20e'):
    files = os.listdir(path)
    num = len(files)//2
    loss=0.0

    for i in range(num):
        img1 = Image.open( path +'/'+'generated_'+str(i)+'.jpg')
        np1 = np.array(img1) /255
        img2 = Image.open( path + '/' + 'target_'+str(i)+'.jpg')
        np2 =np.array( img2 ) /255

        difference = np2 - np1
        l2_norm = np.sum( np.multiply( difference, difference ) ) / element_num
        loss += l2_norm
    
    loss /= num
    print(loss)

evaluate(sys.argv[1])
