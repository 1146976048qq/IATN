import numpy as np
import datetime as dt

start_time = dt.datetime.now()
print(start_time)

ids = np.load('idsMatrix.npy')
print(ids.shape)

maxSeqLength = 250

from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499, 13499) 
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

numDimentions = 50

wordVectors = np.load('wordVectors.npy')

# LSTM mode
import tensorflow as tf

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
test_input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimentions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)


lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
# #  last*weight
prediction = (tf.matmul(last, weight) + bias)  
print("############# prediction ####################", prediction.shape.as_list())


######### import ##############
######### import ##############
######### import ##############
######### import ##############
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# define a standard cross entropy loss with a softmax layer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# visualize the process of loss and accuracy
import datetime
sess = tf.Session()

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("base_models_%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

print(" - - - start_training - - - : ", datetime.datetime.now())
# Train the model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
    # next batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch();
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    # write summary to Tensorboard
    if(i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    # save the network every 10000 training iterations
    if(i % 5000 == 0 and i != 0):
        save_path = saver.save(sess, "base_models/pretrained_lstm.ckpt", global_step=i)
        print("save to %s" % save_path, dt.datetime.now())
writer.close()
print(" - - - end_training - - - : ", datetime.datetime.now())

# # load trained model
# sess = tf.InteractiveSession()
# saver = tf.train.Saver()
# saver.restore(sess, tf.train.latest_checkpoint('models'))   # 自动获取最后一次保存的模型

# Model Test
iterations = 50
for i in range(iterations):
    nextBatch = getTestBatch();
    print("%u batch ---------------------------------------------------- : \n" % i, nextBatch)
    print("%u batch size ---------------------------------------------------- : \n" % i, len(nextBatch))
    # print(" %u batch labels ---------------------------------------- : \n" % i, nextBatchLabels)
    # print(" %u batch labels size ---------------------------------------- : \n" % i, len(nextBatchLabels))
    # print("Accuracy for %u batch:" % i, (sess.run(prediction, {input_data: nextBatch, labels: nextBatchLabels})) * 100, "%")
    print("Accuracy for %u batch:\n" % i, (sess.run(accuracy, {input_data: nextBatch})))




end_time = dt.datetime.now()
print(end_time)
time = (end_time-start_time)
print("LSTM waste time : ", time)