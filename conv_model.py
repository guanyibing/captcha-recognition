import tensorflow as tf
from load_dataset import load_data

max_step=3000
batch_size=50

def variable_summary(var,name):
    with tf.name_scope("summaries"):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean/" + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev/" + name, stddev)

def inference(data_dir,regu,training):
    with tf.variable_scope("input"):
        meta, train_data, test_data = load_data(data_dir)
        CAPATCHA_SIZE=meta["captcha_size"]
        IMAGE_WIDTH=meta["width"]
        IMAGE_HEIGHT=meta["height"]
        IMAGE_SIZE=IMAGE_WIDTH*IMAGE_HEIGHT

    x=tf.placeholder(dtype=tf.float32,shape=[None,IMAGE_HEIGHT,IMAGE_WIDTH])
    y_=tf.placeholder(dtype=tf.float32,shape=[None,CAPATCHA_SIZE])

    input=tf.reshape(x,[-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])
    tf.summary.image("input",input,max_outputs=10)

    with tf.variable_scope("layer1_conv"):
        weight=tf.get_variable("weight",shape=[3,3,1,64],initializer=tf.truncated_normal_initializer())
        bias=tf.get_variable("bias",shape=[64],initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(input,weight,strides=[1,1,1,1],padding="SAME")
        activated_conv=tf.nn.relu(tf.nn.bias_add(conv,bias))#不能直接使用加法，，因为矩阵上不同位置上的节点都要加上同样的偏置项

    with tf.variable_scope("layer2-pool"):
        pool=tf.nn.max_pool(activated_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3_conv"):
        weight = tf.get_variable("weight", shape=[3, 3, 64, 128],initializer= tf.truncated_normal_initializer())
        bias = tf.get_variable("bias", shape=[128], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool, weight, strides=[1,1,1,1], padding="SAME")
        activated_conv = tf.nn.relu(tf.nn.bias_add(conv,bias))

    with tf.variable_scope("layer4-pool"):
        pool = tf.nn.max_pool(activated_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    pool_shape=pool.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool,[-1,nodes])

    with tf.variable_scope("layer5-fc1"):
        weight=tf.get_variable("weight",shape=[nodes,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias=tf.get_variable("bias",shape=[512],initializer=tf.constant_initializer(0.1))
        if regu!=None:
            regularizer=tf.contrib.layers.l2_regularizer(0.001)
            tf.add_to_collection("loss",regularizer(weight))
        fc=tf.nn.relu(tf.matmul(reshaped,weight)+bias)
        if training:
            fc=tf.nn.dropout(fc,keep_prob=0.5)
    with tf.variable_scope("layer6-fc2"):
        weight=tf.get_variable("weight",shape=[512,CAPATCHA_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias=tf.get_variable("bias",shape=[CAPATCHA_SIZE],initializer=tf.constant_initializer(0.1))
        if regu!=None:
            regularizer = tf.contrib.layers.l2_regularizer(0.001)
            tf.add_to_collection("loss",regularizer(weight))
        logit=tf.matmul(fc,weight)+bias

    with tf.variable_scope("loss"):
        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logit))
        loss=cross_entropy+tf.get_collection("loss")
        train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
        variable_summary(loss,"loss")

    with tf.variable_scope("accuracy"):
        pred_correct=tf.equal(tf.argmax(logit,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(pred_correct,tf.float32))
        variable_summary(accuracy,"accuracy")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_writer=tf.summary.FileWriter("./log/train",sess.graph)
        test_writer=tf.summary.FileWriter("./log/test",sess.graph)

        merged=tf.summary.merge_all()
        for i in range(max_step):
            xs,ys=train_data.next_batch(batch_size)
            summary_step,_=sess.run([merged,train_step],feed_dict={x:xs,y_:ys})
            train_writer.add_summary(summary_step,i)
            if i %10==0:
                valid_summary,train_accuracy=sess.run([merged,accuracy],feed_dict={x:xs,y_:ys})
                train_writer.add_summary(valid_summary,i)
                # test_x, test_y = test_data.images,test_data.labels#会报错ResourceExhaustedError，内存不足，虽然测试用理论上是可以用全数据
                test_x, test_y = test_data.next_batch(2000)
                test_summary,test_accuracy=sess.run([merged,accuracy],feed_dict={x:test_x,y_:test_y})
                test_writer.add_summary(test_summary,i)
                print('step %s, training accuracy = %.2f, testing accuracy = %.2f' % (i,train_accuracy ,test_accuracy ))
        train_writer.close()
        test_writer.close()
        test_x,test_y=test_data.next_batch(200)
        test_accuracy=accuracy.eval(feed_dict={x:test_x,y_:test_y})
        print ("test_accuracy:%.2f"%test_accuracy)
def main(argv=None):
    data_dir=".\\images\\2-char-2-groups\\"
    inference(data_dir,True,True)

if __name__=="__main__":
    tf.app.run()
    # parse=argparse.ArgumentParser()
    # parse.add_argument("--data_dir",type=str,default="\\images\\2-char-2-groups\\",help="Directory for input data")
    # parse.add_argument("--train", action="store_true" )
    # parse.add_argument("--regu", action="store_true")
    # FLAGS,unparsered=parse.parse_known_args()
    # tf.app.run(main=inference, argv=[sys.argv])










