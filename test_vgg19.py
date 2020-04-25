"""
Simple tester for the vgg19_trainable
"""

import numpy as np
import tensorflow as tf

import vgg19_trainable as vgg19
import utils

from datagenerator import ImageDataGenerator

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Learning params

# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.01
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                            10, 0.96, staircase=True)
learning_rate = 0.001
num_epochs = 20
batch_size = 32

# Network params
dropout_rate = 0.5
num_classes = 7
train_layers = ['fc8_D', 'fc7_D']
#con_train_layers = ['conv3']

path = 'data.csv'

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, num_classes])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vgg19.npy')
#vgg = vgg19.Vgg19('./test-save.npy')
vgg.build(images, train_mode)
score = vgg.fc8
print(tf.trainable_variables())
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
#con_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in con_train_layers]

with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=true_out))

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    #train_op = optimizer.minimize(loss, var_list=var_list)
    train_op = optimizer.minimize(loss, var_list=tf.trainable_variables())

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(true_out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#gradients = optimizer.compute_gradients(loss, con_var_list)

#saver = tf.train.Saver()
train_generator = ImageDataGenerator(path, horizontal_flip=False, shuffle=True, nb_classes=num_classes)
#val_generator = ImageDataGenerator(val_file, shuffle=False, nb_classes=num_classes)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.train_data_size / batch_size).astype(np.int16)
#val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

with tf.Session() as sess:
    print(train_generator.train_data_size, train_generator.test_data_size)
    sess.run(tf.global_variables_initializer())
    print(vgg.get_var_count())

    training_accuracy=0.0
    for epoch in range(num_epochs):
        step = 0
        while step < train_batches_per_epoch:
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            # _, gradients_val = sess.run([train_op, gradients], feed_dict={images: batch_xs,
            #                             true_out: batch_ys,
            #                             train_mode: True})
            _, training_loss, training_accuracy = sess.run([train_op, loss, accuracy],
                                        feed_dict={images: batch_xs,
                                        true_out: batch_ys,
                                        train_mode: True})
            step = step + 1
            #print(gradients_val[0][0])
            print("Training loss in epoch {}, step {}: {}".format(epoch, step,  round(training_loss,4)))
        print("Training loss in epoch {}: {}".format(epoch, round(training_accuracy, 4)))
        train_generator.reset_train_pointer()
    test_acc = 0.
    test_count = 0
    true_label = []
    pred_label = []
    pred_prob = []
    for _ in range(train_generator.test_data_size):
        batch_tx, batch_ty = train_generator.test_next_batch(1)
        acc = sess.run(accuracy, feed_dict={images: batch_tx,
                                            true_out: batch_ty,
                                            train_mode: False})
        true_label.extend(np.argmax(batch_ty, 1))
        score_P_temp = sess.run(score, feed_dict={images: batch_tx,
                                            true_out: batch_ty,
                                            train_mode: False})
        pred_label.extend(np.argmax(score_P_temp, 1))
        pred_prob.extend([a[1] for a in score_P_temp])
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("Validation Accuracy = {:.4f}".format(test_acc))

    #auc = roc_auc_score(true_label, pred_prob, average='weighted')
    #f1 = f1_score(true_label, pred_label, average='weighted') 
    # test_acc /= test_count
    test_acc = accuracy_score(true_label, pred_label)
    print("Validation Accuracy = {:.4f}".format(test_acc))
    #print("AUC = {:.4f}, F1 = {:.4f}".format(auc, f1))
    #f = open("res.txt", "a")
    #f.write("{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(args.pair, args.split_state, test_acc,auc,f1))
    # Reset the file pointer of the image data generator
    #val_generator.reset_pointer()
    train_generator.reset_train_pointer()

    #res_dict={}
    #res_dict['true_label']=true_label
    #res_dict['pred_label']=pred_label
    #np.save('res.npy',res_dict)
    # test save
    #vgg.save_npy(sess, './test-save.npy')
