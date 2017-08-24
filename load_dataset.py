import os
import numpy as np
from PIL import Image
import gen_captcha
import json

def get_images(filename, flatten,**meta):
    im=Image.open(filename).convert("L")
    data=np.asarray(im)  #np.array在源也是ndarray类型时，仍复制一个副本,而asarray不会
    if flatten:
        data=np.reshape(data,meta["width"]*meta["height"])
    return data

def get_labels(filename, flatten,**meta):
    basename=os.path.basename(filename)
    captcha=basename.split("_")[0]#通过分割文件名得到验证码内容
    captcha_l=list(captcha)
    #将验证码结果映射成one_hot格式
    label=np.zeros(meta["captcha_size"])
    for captcha in captcha_l:
        index=meta["captchas"].index(captcha)
        label[index]=1
    return label

def load_captcha_images(dirname,flatten,ext="png",**meta):
    images=[]
    labels=[]
    for image in os.listdir(dirname):
        if image.endswith(ext):
            filename=os.path.join(dirname,image)
            images.append(get_images(filename,flatten,**meta))
            labels.append(get_labels(filename,flatten,**meta))
    return (np.array(images),np.array(labels))

class DataSet(object):
    def __init__(self,images,labels):
        self.images=images
        self.labels=labels
        self.num_example=images.shape[0]
        self.ptr=0
    @property
    def get_images(self):
        return self.images
    @property
    def get_labels(self):
        return self.labels
    def next_batch(self,size=100,shuffle=True):
        if self.ptr==0:
            if shuffle:
                index=np.arange(self.num_example)
                index=np.random.shuffle(index)
                self.images=self.images[index][0]
                self.labels=self.labels[index][0]
        elif self.ptr+size>self.num_example:
            self.ptr=0
        self.ptr+=size
        return (self.images[self.ptr-size:self.ptr],self.labels[self.ptr-size:self.ptr])

def load_data(data_dir,flatten=False):
    train_dir=os.path.join(data_dir,"train")
    test_dir=os.path.join(data_dir,"test")
    meta_dir=os.path.join(data_dir,"META.json")
    with open(meta_dir,"r") as f:
        meta=json.load(f)
    train_images,train_labels=load_captcha_images(train_dir,flatten,**meta)
    train_data=DataSet(train_images,train_labels)
    test_images,test_labels=load_captcha_images(test_dir,flatten,**meta)
    test_data=DataSet(test_images,test_labels)
    return meta,train_data,test_data
if __name__=="__main__":
    load_data("E:\\Verify Captcha\\images\\2-char-2-groups",flatten=True)