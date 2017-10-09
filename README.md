# captcha-recognition

#### 卷积神经网络识别验证码，运行环境python3.5+tensorflow1.2.1

`代码包括三个部分：`
 * gen_captcha通过python现有的captcha库生成验证码，可以通过参数num_per_image设置验证码位数，n_group设置共生成几组这类验证码

 * load_data将生成的验证码图片依次读入，存入一个数组中；验证码图片的结果在所有组成字符列表中映射为one_hot格式，比如验证码是2和7，而组成字符列表是[0,1,2,3,4,5,6,7,8,9],结果就是[0,0,1,0,0,0,1,0,0]

 * 卷积网络模型参照LeNet-5模型，注意卷积层加偏置项，要用tf.nn.bias_add函数，不能直接用+号，因为矩阵上不同位置上的节点都要加上同样的偏置项```tf.nn.bias_add(conv,bias)```；池化层采用max_pool最大池化层。

 * 通过tf.summary模块，将计算图的结构和信息存入日志，然后在terminal输入tensroboard --logdir=="/.../",在浏览器打开localhost:6006查看可视化结果。
