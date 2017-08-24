import string
import os
import shutil
from captcha.image import ImageCaptcha
import itertools
import uuid
import json
import argparse

FLAGS=None

def gen_captcha_list():
    cate_map=[(map(str,range(10))),(string.ascii_lowercase),(string.ascii_uppercase)]
    return [i for captchas in cate_map for i in captchas]

def gen_captcha(img_dir,width,height,num_per_image,n_groups,captcha_list):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    image=ImageCaptcha(width=width,height=height)
    print ("generating %d groups in %s"%(n_groups,img_dir))
    for _ in range(n_groups):
        for i in itertools.permutations(captcha_list,num_per_image):#每num_per_image个字符有序组合成验证码内容
            captcha="".join(i)
            file_name=os.path.join(img_dir,"%s_%s.png"%(captcha,uuid.uuid4()))
            image.write(captcha,file_name)#生成验证码

def gen_dataset(root_dir):
    n_groups_train=FLAGS.n
    n_groups_test=max(n_groups_train*FLAGS.t,1)
    num_per_image=FLAGS.npi
    width=80+20*num_per_image
    height=60
    def build_path(x):
        return os.path.join(root_dir,"%s-char-%s-groups"%(num_per_image,n_groups_train),x)
    captcha_list=gen_captcha_list()
    gen_captcha(build_path("train"),width,height,num_per_image,n_groups_train,captcha_list)
    gen_captcha(build_path("test"),width,height,num_per_image,n_groups_test,captcha_list)
    meta = {
        'num_per_image': num_per_image,
        'captcha_size': len(captcha_list),
        'captchas': ''.join(captcha_list),
        'n_train': n_groups_train,
        'n_test': n_groups_test,
        'width': width,
        'height': height,
    }
    meta_filename=build_path("meta.json")
    with open(meta_filename,"w") as f:
        json.dump(meta,f,indent=4)

if __name__=="__main__":
    parse=argparse.ArgumentParser()
    parse.add_argument("-n",default=2,type=int,help="n groups of captcha permutations")
    parse.add_argument("-t",default=0.2,type=float,help="ratio of n_test_groups/n_train_groups")
    parse.add_argument("-npi",default=2,type=int,help="num of char per captcha image")
    FLAGS,unparsed=parse.parse_known_args()

    gen_dataset("images")


