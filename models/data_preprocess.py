# encoding=UTF-8
import numpy as np
import matplotlib as plt
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# %%
def data_preprocess(path='/home/xinsheng/yizhehang/Research/Chinese_Font_Transfer/Font2img/img_lib',
                    source_font='heiti',
                    target_font='lixuke'):
    """
        normalize font images to [0,1] and then create the source-target font pairs
    """
    # load the source fonts and target fonts
    source_path = path+'/'+source_font
    source_fonts = list()
    target_path = path + '/' + target_font
    target_fonts = list()
    files = os.listdir(source_path)
    mid1 = np.zeros((64, 64, 1))
    mid2 = np.zeros((64, 64, 1))   

    
    for file in files:
        #load a source font
        img1 = Image.open(source_path+'/'+file)
        np1 = np.array(img1)
        mid1 = np1[:, :, 0]
        mid1 = mid1[:,:,np.newaxis]
        source_fonts.append(mid1)

        #load a target font
        img2 = Image.open(target_path + '/'+file)
        np2 = np.array(img2)
        mid2 = np2[:, :, 0]
        mid2 = mid2[:,:,np.newaxis]
        target_fonts.append(mid2)

    source_fonts = np.array(source_fonts)
    source_fonts = source_fonts.astype(np.float32)
    source_fonts /= 255

    target_fonts = np.array(target_fonts)
    target_fonts = target_fonts.astype(np.float32)
    target_fonts /= 255


    return source_fonts, target_fonts


'''img = Image.fromarray(  a[0][:128][:128][0] )
img.show()

b = np.zeros((128,128))
a[:,:,0].shape
a *= 255
img = Image.fromarray(  a )
img.show()'''

# %%


class Dataset:
    def __init__(self, source_fonts, target_fonts, test_frac=0.4, val_frac=0.1, scale_func=None):
        """
            create the training set, validation set and testing set
        """
        self.data_num = int(len(source_fonts))
        self.val_num = int(self.data_num*(1-test_frac)*val_frac)
        self.train_num = int(self.data_num * (1 - test_frac)) - self.val_num
        self.test_num = self.data_num-self.val_num-self.train_num

        self.train = {}
        self.test = {}
        self.valid = {}


        """self.train['source_font'] = np.rollaxis(
            source_fonts[:self.train_num], axis=3)
        self.valid['source_font'] = np.rollaxis(
            source_fonts[self.train_num: self.train_num+self.val_num], axis=3)
        self.test['source_font'] = np.rollaxis(
            source_fonts[self.train_num + self.val_num:], axis=3)

        self.train['target_font'] = np.rollaxis(
            target_fonts[:self.train_num], axis=3)
        self.valid['target_font'] = np.rollaxis(
            target_fonts[self.train_num: self.train_num+self.val_num], axis=3)
        self.test['target_font'] = np.rollaxis(
            target_fonts[self.train_num + self.val_num:], axis=3)"""

        self.train['source_font'] =source_fonts[:self.train_num]
        self.valid['source_font'] =source_fonts[self.train_num: self.train_num+self.val_num]
        self.test['source_font'] = source_fonts[self.train_num + self.val_num:]

        self.train['target_font'] =target_fonts[:self.train_num]
        self.valid['target_font'] =target_fonts[self.train_num: self.train_num+self.val_num]
        self.test['target_font'] =target_fonts[self.train_num + self.val_num:]

    def shuffle_data( self ):
        idx = np.arange(self.train_num)
        np.random.shuffle(idx)
        self.train['source_font'], self.train['target_font'] = self.train['source_font'][idx], self.train['target_font'][idx]

    def get_batches(self, batch_size):
        """
            generate one batch of data
        """
        batch_num = self.train_num // batch_size

        for ii in range(0, batch_num * batch_size, batch_size):
            source_font = self.train['source_font'][ii:ii+batch_size]
            target_font = self.train['target_font'][ii:ii+batch_size]

            yield source_font, target_font


# %%
#source_fonts, target_fonts = data_preprocess()


# %%

#dataset = Dataset( source_fonts, target_fonts)
#print(dataset.train['source_font'].shape)

# 解决getbatches报错：'KeyValueError target_font'
# %%
