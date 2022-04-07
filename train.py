import os, sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model
from utils import DataInput, compute_auc, compute_hr, dictionary

#Note: this code must be run using tensorflow 1.4.0
tf.reset_default_graph()
#Data Loading

data1 = pd.read_csv('amazon_toys_filter.csv')
data2 = pd.read_csv('amazon_videogames_filter.csv')
data1.columns = ['utdid','vdo_id','click','hour']
data2.columns = ['utdid','vdo_id','click','hour']

data = pd.concat([data1,data2])
user_id = data[['utdid']].drop_duplicates().reindex()
user_id['user_id'] = np.arange(len(user_id))
data1 = pd.merge(data1, user_id, on=['utdid'], how='left')
data2 = pd.merge(data2, user_id, on=['utdid'], how='left')
item_id_1 = data1[['vdo_id']].drop_duplicates().reindex()
item_id_1['video_id'] = np.arange(len(item_id_1))
data1 = pd.merge(data1, item_id_1, on=['vdo_id'], how='left')
item_id_2 = data2[['vdo_id']].drop_duplicates().reindex()
item_id_2['video_id'] = np.arange(len(item_id_2))
data2 = pd.merge(data2, item_id_2, on=['vdo_id'], how='left')
data1 = data1[['user_id','video_id','click','hour']]
data2 = data2[['user_id','video_id','click','hour']]
user_count = len(user_id)
item_count_1 = len(item_id_1)
item_count_2 = len(item_id_2)
item_count = max(item_count_1, item_count_2)

#Model Training
batch_size = 32
lr = 1e-2

#trainset1, testset1, trainset2, testset2 = train_test_set(data1, data2, batch_size)
validate = 4 * len(data1) // 5
train_data1 = data1.loc[:validate,]
test_data1 = data1.loc[validate:,]
validate = 4 * len(data2) // 5
train_data2 = data2.loc[:validate,]
test_data2 = data2.loc[validate:,]
trainset1, testset1, trainset2, testset2 = [], [], [], []

userid = list(set(user_id['user_id']))
for user in userid:
    train_user1 = train_data1.loc[train_data1['user_id']==user]
    train_user1 = train_user1.sort_values(['hour'])
    length1 = len(train_user1)
    train_user1.index = range(length1)
    train_user2 = train_data2.loc[train_data2['user_id']==user]
    train_user2 = train_user2.sort_values(['hour'])
    length2 = len(train_user2)
    train_user2.index = range(length2)
    length = min(length1, length2)
    if length > 10:
        for i in range(length-10):
            trainset1.append((train_user1.loc[i+9,'user_id'], list(train_user1.loc[i:i+9,'video_id']), list(train_user2.loc[i:i+9,'video_id']), train_user1.loc[i+9,'video_id'], float(train_user1.loc[i+9,'click'])))
            trainset2.append((train_user2.loc[i+9,'user_id'], list(train_user2.loc[i:i+9,'video_id']), list(train_user1.loc[i:i+9,'video_id']), train_user2.loc[i+9,'video_id'], float(train_user2.loc[i+9,'click'])))
    test_user1 = test_data1.loc[test_data1['user_id']==user]
    test_user1 = test_user1.sort_values(['hour'])
    length1 = len(test_user1)
    test_user1.index = range(length1)
    test_user2 = test_data2.loc[test_data2['user_id']==user]
    test_user2 = test_user2.sort_values(['hour'])
    length2 = len(test_user2)
    test_user2.index = range(length2)
    length = min(length1, length2)
    if length > 10:
        for i in range(length-10):
            testset1.append((test_user1.loc[i+9,'user_id'], list(test_user1.loc[i:i+9,'video_id']), list(test_user2.loc[i:i+9,'video_id']), test_user1.loc[i+9,'video_id'], float(test_user1.loc[i+9,'click'])))
            testset2.append((test_user2.loc[i+9,'user_id'], list(test_user2.loc[i:i+9,'video_id']), list(test_user1.loc[i:i+9,'video_id']), test_user2.loc[i+9,'video_id'], float(test_user2.loc[i+9,'click'])))
random.shuffle(trainset1)
random.shuffle(testset1)
random.shuffle(trainset2)
random.shuffle(testset2)
trainset1 = trainset1[:len(trainset1)//batch_size*batch_size]
testset1 = testset1[:len(testset1)//batch_size*batch_size]
trainset2 = trainset2[:len(trainset2)//batch_size*batch_size]
testset2 = testset2[:len(testset2)//batch_size*batch_size]

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Model(user_count, item_count, batch_size)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    print('Domain_A_Initialized_AUC: %.4f\tDomain_B_Initialized_AUC: %.4f' % compute_auc(sess, model, testset1, testset2))
    sys.stdout.flush()
    start_time = time.time()
    last_auc = 0.0
    
    for _ in range(1000):
        loss_sum = 0.0
        for uij in DataInput(trainset1, batch_size):
            loss = model.train_1(sess, uij, lr)
            loss_sum += loss
        for uij in DataInput(trainset2, batch_size):
            loss = model.train_2(sess, uij, lr)
            loss_sum += loss
        model.train_orth(sess, uij[0], lr)
        test_auc_1, test_auc_2 = compute_auc(sess, model, testset1, testset2)
        train_auc_1, train_auc_2 = compute_auc(sess, model, trainset1, trainset2)
        print('Epoch %d \tDomain A Train_AUC: %.4f\tTest_AUC: %.4F' % (model.global_epoch_step.eval(), train_auc_1, test_auc_1))
        print('Epoch %d \tDomain B Train_AUC: %.4f\tTest_AUC: %.4F' % (model.global_epoch_step.eval(), train_auc_2, test_auc_2))
        print('Epoch %d \tTrain_loss: %.4f' % (model.global_epoch_step.eval(), loss_sum))
        print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time()-start_time))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
        hit_1, hit_2 = compute_hr(sess, model, testset1, testset2)
        print('Epoch %d Domain A_Hit_Rate: %.4f\tDomain B_Hit_Rate: %.4f\t' % (model.global_epoch_step.eval(), hit_1, hit_2))
        sys.stdout.flush()
