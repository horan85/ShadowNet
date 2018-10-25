from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime
import random
import cv2
import os
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1" # select GPU 
    

# set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
now = datetime.datetime.now()
dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
flags.DEFINE_string('summary_dir', 'logs/combined/equal/{}'.format(dt), 'Summaries directory')
# # if summary directory exist, delete the previous summaries
# # if tf.gfile.Exists(FLAGS.summary_dir):
# #	 tf.gfile.DeleteRecursively(FLAGS.summary_dir)
# #	 tf.gfile.MakeDirs(FLAGS.summary_dir)



# from cifar-10 webpage + own mods to get 50k train and 10k test images into 1-1 batches
def unpickle(name="train"):
    import pickle
    imagearray = None

    if name is "train":
    	for i in range(5):
    		with open('cifar10-data/cifar_10/data_batch_'+ str(i+1), 'rb') as f:
        		dict = pickle.load(f, encoding='bytes')
        		images = dict[b'data']
       			labels = dict[b'labels']
        		labelarraypart = np.zeros(shape=(len(labels),10))
        		for l in range(len(labels)):
        			labelarraypart[l,:]=one_hot_enc(labels[l],10)
        		images = np.reshape(images,(-1,3,32,32)) # -1 since we do not know the size
        		images = images.transpose([0,2,3,1])
        		# resize could be here

        		imagearraypart = np.array(images)

        		if imagearray is None:
        			imagearray = imagearraypart
        			labelarray = labelarraypart
        		else:
        			imagearray = np.concatenate((imagearray,imagearraypart))
        			labelarray = np.concatenate((labelarray,labelarraypart))
    elif name is "test":
	   	with open('cifar10-data/cifar_10/test_batch', 'rb') as f:
	   		dict = pickle.load(f, encoding='bytes')
	   		images = dict[b'data']
	   		labels = dict[b'labels']
	   		labelarray = np.zeros(shape=(len(labels),10))
	   		for l in range(len(labels)):
	   			labelarray[l,:]=one_hot_enc(labels[l],10)
	   		images = np.reshape(images,(-1,3,32,32)) # -1 since we do not know the size
	   		images = images.transpose([0,2,3,1])
	   		imagearray = np.array(images)

    return imagearray,labelarray

def one_hot_enc(label,num_classes):
    one_hot_encoded_label = np.zeros(num_classes)
    one_hot_encoded_label[label] = 1
    return one_hot_encoded_label

def resize_batch(InputBatch,ImgSize):
    # Resize Batch and Train Data
    BatchLength = len(InputBatch)
    OutBatch = np.zeros((BatchLength, ImgSize[0], ImgSize[1], ImgSize[2]))
    for i in range(BatchLength):
        OutBatch[i, :, :, :] = cv2.resize(InputBatch[i, :, :, :], (ImgSize[0], ImgSize[1]))
    return OutBatch

def filter_and_convert_mnist_data(AllData,AllLabels):
    ## something like this should also work
    # MnistVals = [0,1]
    # Idxs = []
    # for vals in MnistVals:
    #    Idxs = np.concatenate((Idxs,np.where(AllLabels==vals)[0]))
    Idxs0 = np.where(AllLabels==0)[0]
    Idxs1 = np.where(AllLabels==1)[0]
    Idxs = np.concatenate((Idxs0,Idxs1))
    Idxs = np.sort(Idxs)
    # expand dims and RGB conversion
    FilteredData = np.expand_dims(AllData[Idxs,:,:],-1)
    FilteredDataRGB = np.tile(FilteredData, [1,1,1,3])
    # convert labels to one_hot
    FilteredLabelsRaw = AllLabels[Idxs]
    FilteredLabels = np.zeros(shape=(len(FilteredLabelsRaw),2))
    for l in range(len(FilteredLabelsRaw)):
        FilteredLabels[l,:]=one_hot_enc(FilteredLabelsRaw[l],2)
    return FilteredDataRGB,FilteredLabels


# Getting CIFAR DATA
TrainData,TrainLabels = unpickle()
TestData,TestLabels = unpickle("test")

print('test labels : ',len(TestLabels))
print('test data   : ',len(TestData))
print('train labels: ',len(TrainLabels))
print('train data  : ',len(TrainData))
print(TrainData.shape)


# get MNIST DATA
print('Getting MNIST data...')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist-data/", one_hot=False) # one_hot false for easier filtering

MnistAllTrainData = np.reshape(mnist.train.images,[-1,28,28])
MnistAllTrainLabels = mnist.train.labels
MnistAllTestData = np.reshape(mnist.test.images,[-1,28,28])
MnistAllTestLabels = mnist.test.labels
print('MNIST is loaded')

print('MNIST data :',MnistAllTrainData.shape)
print('MNIST label :',MnistAllTrainLabels.shape)


## Filtering MNIST Data - V2
mnist_TrainData, mnist_TrainLabels = filter_and_convert_mnist_data(MnistAllTrainData, MnistAllTrainLabels)
mnist_TestData, mnist_TestLabels = filter_and_convert_mnist_data(MnistAllTestData, MnistAllTestLabels)

print('###############################')
print('mnist train data shape: ', mnist_TrainData.shape )
print('mnist test data shape: ', mnist_TestData.shape )
print('###############################')


# # # # Resize Data
# ImgSize = [227, 227, 3]  # Input img will be resized to this size
# # Resize MNIST Data
# mnist_TrainDataRS = resize_batch(MnistFiltTrainDataRGB, ImgSize)
# mnist_TestDataRS = resize_batch(MnistFiltTestDataRGB, ImgSize)
CommonImageSize = [32, 32, 3]
mnist_TrainDataRS = resize_batch(mnist_TrainData, CommonImageSize)
mnist_TestDataRS = resize_batch(mnist_TestData, CommonImageSize)

print('###############################')
print('mnist train data shape: ', mnist_TrainDataRS.shape )
print('mnist test data shape: ', mnist_TestDataRS.shape )
print('###############################')

# # Resize Cifar Data
# cifar_TrainDataRS = resize_batch(TrainData, ImgSize)
# cifar_TestDataRS = resize_batch(TestData, ImgSize)

# # TestData = cifar_TestData
# # TrainData = cifar_TrainData
# print('##############################################')
# print('mnist train data shape : ', mnist_TrainDataRS.shape)
# print('cifar train data shape : ', cifar_TrainDataRS.shape)
# print('##############################################')


# Set parameters

BatchLength = 64  # 32 images are in a minibatch
ImgSize = [227, 227, 3]  # Input img will be resized to this size
NumIteration = 8001
mnist_maxIteration = 6001
LearningRate = 1e-4  # learning rate of the algorithm
cifar_NumClasses = 10  # number of output classes
Dropout = 0.5  # droupout parameters in the FNN layer - currently not used
EvalFreq = 500  # evaluate on every 100th iteration
mnist_PostValidFreq = 10
mnist_NumClasses = 2
UseBatchNorm = False

# resultsFile = open("minst20th-furthertrain_"+str(DROPOUT)+"results.csv",'w')
# resultsFile.write("Iteration;Cifar Accuracy;Cifar Loss;Cifar Validation;Cifar VLoss; Mnist Accuracy; Mnist Validation")
# resultsFile.write("\n")


# Make a single MNIST and CIFAR Batch for VALIDATION:
if UseBatchNorm:
    cifar_TestIdxs = random.sample(range(TrainData.shape[0]), BatchLength)
    cifar_ValidationData = TrainData[cifar_TestIdxs, :, :, :]
    # cifar_ValidationData = resize_batch(cifar_ValidationData,ImgSize)

    mnist_TestIdxs = random.sample(range(mnist_TrainDataRS.shape[0]), BatchLength)
    mnist_ValidationData = mnist_TrainDataRS[mnist_TestIdxs, :, :, :]
    # mnist_ValidationData = resize_batch(mnist_ValidationData,ImgSize)


InputData = tf.placeholder(tf.float32,[None,CommonImageSize[0],CommonImageSize[1],CommonImageSize[2]]) # network input
# cifar_InputData = tf.placeholder(tf.float32,[None,32,32,3]) # network input
# mnist_InputData = tf.placeholder(tf.float32,[None,28,28,3]) # network input
# cifar_InputDataR = tf.image.resize_images(cifar_InputData,(227,227))
# mnist_InputDataR = tf.image.resize_images(mnist_InputData,(227,227))
# InputData_resized =  tf.image.resize_images(InputData,(ImgSize[0],ImgSize[1]))
InputData_resized =  InputData
cifar_InputLabels = tf.placeholder(tf.int32, [None,cifar_NumClasses])  # desired network output
cifar_OneHotLabels = cifar_InputLabels
mnist_InputLabels = tf.placeholder(tf.int32, [None,mnist_NumClasses])  # desired network output
mnist_OneHotLabels = mnist_InputLabels

KeepProb = tf.placeholder(tf.float32)  # dropout (keep probability)

print('input placeholder size: ', InputData.shape)
print('label placeholder size: ',cifar_InputLabels.shape)
print('onehotlabel size: ',cifar_OneHotLabels.shape)


# Data augmentation
# for ind in range(Data.shape[0]):
#     Data[ind ,:,:,:]-=np.mean(Data[ind ,:,:,:])
#     Data[ind ,:,:,:]/=np.var(Data[ind ,:,:,:])
#     #data augmentation on training set
#     #add data augmentation: zero padding and random cut
#     Padded=np.pad(Data[ind ,:,:,:], ((3,3),(3,3,),(0,0)),'constant', constant_values=(0, 0))
#     XStart=np.random.randint(7)
#     YStart=np.random.randint(7)
#     Data[ind ,:,:,:]=Padded[XStart:(XStart+Size[0]),YStart:(YStart+Size[1]),:]
#     #add horizontal flip
#     if np.random.randint(2)==0:
#         Data[ind ,:,:,:]=np.fliplr(Data[ind ,:,:,:])



def MakeAlexNet(Input, Size, KeepProb):
    CurrentInput = Input  # 227,227,3
    CurrentInput = CurrentInput /255.0
    with tf.variable_scope('shared_conv1'):
        # first convolution
        W = tf.get_variable('W', [11, 11, 3, 96])
        Bias = tf.get_variable(
            'Bias', [96], initializer=tf.constant_initializer(0.1))
        ConvResult1 = tf.nn.conv2d(CurrentInput, W, strides=[
                                   1, 4, 4, 1], padding='SAME')  # VALID, SAME
        if UseBatchNorm:
            # BatchNorm
            NumKernel = 96
            beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))     # itt NumKernel az adott retegben talahlato feature-ok/csatornak szama
            gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))   # itt is hasonlo az elozohoz
            Mean,Variance = tf.nn.moments(ConvResult1,[0,1,2])    # ConvResult az adat, amit normalizalni szeretnenk (ezt altalaban a convolucio utan szoktak tenni, s igy nincs szukseg kulon bias-ra)
            PostNormalized = tf.nn.batch_normalization(ConvResult1,Mean,Variance,beta,gamma,1e-10)   # PostNormalized pedig a mar batchnormalizalt ertekeket.
            ConvResult1 = PostNormalized
        else:
            ConvResult1 = tf.add(ConvResult1, Bias)

        # first relu
        ReLU1 = tf.nn.relu(ConvResult1)
        # response normalization
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        Norm1 = tf.nn.local_response_normalization(
            ReLU1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
        # first pooling
        Pool1 = tf.nn.max_pool(Norm1, ksize=[1, 3, 3, 1], strides=[
                               1, 2, 2, 1], padding='VALID')
    with tf.variable_scope('shared_conv2'):
        # second convolution
        W = tf.get_variable('W', [5, 5, 96, 256])
        Bias = tf.get_variable(
            'Bias', [256], initializer=tf.constant_initializer(0.1))
        ConvResult2 = tf.nn.conv2d(
            Pool1, W, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME
        if UseBatchNorm:
            # BatchNorm
            NumKernel = 256
            beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))     # itt NumKernel az adott retegben talahlato feature-ok/csatornak szama
            gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))   # itt is hasonlo az elozohoz
            Mean,Variance = tf.nn.moments(ConvResult2,[0,1,2])    # ConvResult az adat, amit normalizalni szeretnenk (ezt altalaban a convolucio utan szoktak tenni, s igy nincs szukseg kulon bias-ra)
            PostNormalized = tf.nn.batch_normalization(ConvResult2,Mean,Variance,beta,gamma,1e-10)   # PostNormalized pedig a mar batchnormalizalt ertekeket.
            ConvResult2 = PostNormalized
        else:
            ConvResult2 = tf.add(ConvResult2, Bias) # commented because of BatchNorm

        # second relu
        ReLU2 = tf.nn.relu(ConvResult2)
        # response normalization
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        Norm2 = tf.nn.local_response_normalization(
            ReLU2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
        # second pooling
        Pool2 = tf.nn.max_pool(Norm2, ksize=[1, 3, 3, 1], strides=[
                               1, 2, 2, 1], padding='VALID')
    with tf.variable_scope('shared_conv3'):
        # third convolution
        W = tf.get_variable('W', [3, 3, 256, 384])
        Bias = tf.get_variable(
            'Bias', [384], initializer=tf.constant_initializer(0.1))
        ConvResult3 = tf.nn.conv2d(
            Pool2, W, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME

        if UseBatchNorm:
            # BatchNorm
            NumKernel = 384
            beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))     # itt NumKernel az adott retegben talahlato feature-ok/csatornak szama
            gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))   # itt is hasonlo az elozohoz
            Mean,Variance = tf.nn.moments(ConvResult3,[0,1,2])    # ConvResult az adat, amit normalizalni szeretnenk (ezt altalaban a convolucio utan szoktak tenni, s igy nincs szukseg kulon bias-ra)
            PostNormalized = tf.nn.batch_normalization(ConvResult3,Mean,Variance,beta,gamma,1e-10)   # PostNormalized pedig a mar batchnormalizalt ertekeket.
            ConvResult3 = PostNormalized
        else:
            ConvResult3 = tf.add(ConvResult3, Bias)

        # third relu
        ReLU3 = tf.nn.relu(ConvResult3)
    with tf.variable_scope('shared_conv4'):
        # fourth convolution
        W = tf.get_variable('W', [3, 3, 384, 384])
        Bias = tf.get_variable(
            'Bias', [384], initializer=tf.constant_initializer(0.1))
        ConvResult4 = tf.nn.conv2d(
            ReLU3, W, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME

        if UseBatchNorm:
            # BatchNorm
            NumKernel = 384
            beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))     # itt NumKernel az adott retegben talahlato feature-ok/csatornak szama
            gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))   # itt is hasonlo az elozohoz
            Mean,Variance = tf.nn.moments(ConvResult4,[0,1,2])    # ConvResult az adat, amit normalizalni szeretnenk (ezt altalaban a convolucio utan szoktak tenni, s igy nincs szukseg kulon bias-ra)
            PostNormalized = tf.nn.batch_normalization(ConvResult4,Mean,Variance,beta,gamma,1e-10)   # PostNormalized pedig a mar batchnormalizalt ertekeket.
            ConvResult4 = PostNormalized
        else:
            ConvResult4 = tf.add(ConvResult4, Bias)

        # fourth relu
        ReLU4 = tf.nn.relu(ConvResult4)
    with tf.variable_scope('shared_conv5'):
        # fifth convolution
        W = tf.get_variable('W', [3, 3, 384, 256])
        Bias = tf.get_variable(
            'Bias', [256], initializer=tf.constant_initializer(0.1))
        ConvResult5 = tf.nn.conv2d(
            ReLU4, W, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME

        if UseBatchNorm:
            # BatchNorm
            NumKernel = 256
            beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))     # itt NumKernel az adott retegben talahlato feature-ok/csatornak szama
            gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))   # itt is hasonlo az elozohoz
            Mean,Variance = tf.nn.moments(ConvResult5,[0,1,2])    # ConvResult az adat, amit normalizalni szeretnenk (ezt altalaban a convolucio utan szoktak tenni, s igy nincs szukseg kulon bias-ra)
            PostNormalized = tf.nn.batch_normalization(ConvResult5,Mean,Variance,beta,gamma,1e-10)   # PostNormalized pedig a mar batchnormalizalt ertekeket.
            ConvResult5 = PostNormalized
        else:
            ConvResult5 = tf.add(ConvResult5, Bias)

        # fifth relu
        ReLU5 = tf.nn.relu(ConvResult5)
        # fifth pooling
        Pool5 = tf.nn.max_pool(ReLU5, ksize=[1, 3, 3, 1], strides=[
                               1, 2, 2, 1], padding='VALID')
    with tf.variable_scope('shared_FC1'):
        # first Fully-connected layer
        CurrentShape = Pool5.get_shape()
        FeatureLength = int(
            CurrentShape[1] * CurrentShape[2] * CurrentShape[3])
        FC = tf.reshape(Pool5, [-1, FeatureLength])
        W = tf.get_variable('W', [FeatureLength, 4096])
        FC = tf.matmul(FC, W)
        Bias = tf.get_variable(
            'Bias', [4096], initializer=tf.constant_initializer(0.1))
        FC = tf.add(FC, Bias)
        # relu
        FCReLU1 = tf.nn.relu(FC)
    with tf.variable_scope('shared_FC2'):
        # first Fully-connected layer
        FC = tf.reshape(FCReLU1, [-1, 4096])
        W = tf.get_variable('W', [4096, 4096])
        FC = tf.matmul(FC, W)
        Bias = tf.get_variable(
            'Bias', [4096], initializer=tf.constant_initializer(0.1))
        FC = tf.add(FC, Bias)
        # relu
        FC = tf.nn.dropout(FC, KeepProb)
        FCReLU2 =  tf.nn.relu(FC)
        #FCReLU2 = AddRelUfc(FC)
    with tf.variable_scope('cifar_FC3'):
        # first Fully-connected layer
        FC = tf.reshape(FCReLU2, [-1, 4096])
        W = tf.get_variable('W', [4096, cifar_NumClasses])
        FC = tf.matmul(FC, W)
        Bias = tf.get_variable(
            'Bias', [cifar_NumClasses], initializer=tf.constant_initializer(0.1))
        FC = tf.add(FC, Bias)
        # no relu at the end
        cifar_Out = FC

    with tf.variable_scope('mnist_FC3'):
        # first Fully-connected layer
        FC = tf.reshape(FCReLU2, [-1, 4096])
        W = tf.get_variable('W', [4096, mnist_NumClasses])
        FC = tf.matmul(FC, W)
        Bias = tf.get_variable(
            'Bias', [mnist_NumClasses], initializer=tf.constant_initializer(0.1))
        FC = tf.add(FC, Bias)
        # no relu at the end
        mnist_Out = FC

    return cifar_Out, mnist_Out

def MakeSmallAlexNet(Input, Size, KeepProb):
    CurrentInput = Input  # 227,227,3
    CurrentInput = CurrentInput /255.0
    with tf.variable_scope('shared_conv1'):
        # first convolution
        W = tf.get_variable('W', [5, 5, 3, 64])
        Bias = tf.get_variable(
            'Bias', [64], initializer=tf.constant_initializer(0.0))
        ConvResult1 = tf.nn.conv2d(CurrentInput, W, strides=[
                                   1, 1, 1, 1], padding='SAME')  # VALID, SAME
        if UseBatchNorm:
            # BatchNorm
            NumKernel = 96
            beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))     # itt NumKernel az adott retegben talahlato feature-ok/csatornak szama
            gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))   # itt is hasonlo az elozohoz
            Mean,Variance = tf.nn.moments(ConvResult1,[0,1,2])    # ConvResult az adat, amit normalizalni szeretnenk (ezt altalaban a convolucio utan szoktak tenni, s igy nincs szukseg kulon bias-ra)
            PostNormalized = tf.nn.batch_normalization(ConvResult1,Mean,Variance,beta,gamma,1e-10)   # PostNormalized pedig a mar batchnormalizalt ertekeket.
            ConvResult1 = PostNormalized
        else:
            ConvResult1 = tf.add(ConvResult1, Bias)

        # first relu
        ReLU1 = tf.nn.relu(ConvResult1)
        Pool1 = tf.nn.max_pool(ReLU1, ksize=[1, 3, 3, 1], strides=[
                               1, 2, 2, 1], padding='VALID')
        # response normalization
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        Norm1 = tf.nn.local_response_normalization(
            Pool1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
        # first pooling
    with tf.variable_scope('shared_conv2'):
        # second convolution
        W = tf.get_variable('W', [5, 5, 64, 64])
        Bias = tf.get_variable(
            'Bias', [64], initializer=tf.constant_initializer(0.1))
        ConvResult2 = tf.nn.conv2d(
            Pool1, W, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME
        if UseBatchNorm:
            # BatchNorm
            NumKernel = 256
            beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))     # itt NumKernel az adott retegben talahlato feature-ok/csatornak szama
            gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))   # itt is hasonlo az elozohoz
            Mean,Variance = tf.nn.moments(ConvResult2,[0,1,2])    # ConvResult az adat, amit normalizalni szeretnenk (ezt altalaban a convolucio utan szoktak tenni, s igy nincs szukseg kulon bias-ra)
            PostNormalized = tf.nn.batch_normalization(ConvResult2,Mean,Variance,beta,gamma,1e-10)   # PostNormalized pedig a mar batchnormalizalt ertekeket.
            ConvResult2 = PostNormalized
        else:
            ConvResult2 = tf.add(ConvResult2, Bias) # commented because of BatchNorm

        # second relu
        ReLU2 = tf.nn.relu(ConvResult2)
        # response normalization
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        Norm2 = tf.nn.local_response_normalization(
            ReLU2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
        # second pooling
        Pool2 = tf.nn.max_pool(Norm2, ksize=[1, 3, 3, 1], strides=[
                               1, 2, 2, 1], padding='VALID')
    
    with tf.variable_scope('shared_FC1'):
        # first Fully-connected layer
        CurrentShape = Pool2.get_shape()
        FeatureLength = int(
            CurrentShape[1] * CurrentShape[2] * CurrentShape[3])
        FC = tf.reshape(Pool2, [-1, FeatureLength])
        W = tf.get_variable('W', [FeatureLength, 384])
        FC = tf.matmul(FC, W)
        Bias = tf.get_variable(
            'Bias', [384], initializer=tf.constant_initializer(0.1))
        FC = tf.add(FC, Bias)
        # relu
        FCReLU1 = tf.nn.relu(FC)
    with tf.variable_scope('shared_FC2'):
        # first Fully-connected layer
        FC = FCReLU1#tf.reshape(FCReLU1, [-1, 192])
        W = tf.get_variable('W', [384, 192])
        FC = tf.matmul(FC, W)
        Bias = tf.get_variable(
            'Bias', [192], initializer=tf.constant_initializer(0.1))
        FC = tf.add(FC, Bias)
        # relu
        FC = tf.nn.dropout(FC, KeepProb)
        FCReLU2 =  tf.nn.relu(FC)
        #FCReLU2 = AddRelUfc(FC)
    with tf.variable_scope('cifar_FC3'):
        # first Fully-connected layer
        FC = tf.reshape(FCReLU2, [-1, 192])
        W = tf.get_variable('W', [192, cifar_NumClasses])
        FC = tf.matmul(FC, W)
        Bias = tf.get_variable(
            'Bias', [cifar_NumClasses], initializer=tf.constant_initializer(0.0))
        FC = tf.add(FC, Bias)
        # no relu at the end
        cifar_Out = FC
    with tf.variable_scope('mnist_FC3'):
        # first Fully-connected layer
        FC = tf.reshape(FCReLU2, [-1, 192])
        W = tf.get_variable('W', [192, mnist_NumClasses])
        FC = tf.matmul(FC, W)
        Bias = tf.get_variable(
            'Bias', [mnist_NumClasses], initializer=tf.constant_initializer(0.0))
        FC = tf.add(FC, Bias)
        # no relu at the end
        mnist_Out = FC

    return cifar_Out, mnist_Out

# Construct model
cifar_PredWeights, mnist_PredWeights = MakeSmallAlexNet(InputData_resized, ImgSize, KeepProb)


# Get variables
train_vars = tf.trainable_variables()
shared_vars = [var for var in train_vars if 'shared_' in var.name]
cifar_vars = [var for var in train_vars if 'cifar_' in var.name]
mnist_vars = [var for var in train_vars if 'mnist_' in var.name]


# Define loss and optimizer
with tf.name_scope('loss'):
    cifar_Loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(cifar_OneHotLabels, cifar_PredWeights))
    mnist_Loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(mnist_OneHotLabels, mnist_PredWeights))
    #combined_Loss = tf.add(cifar_Loss,mnist_Loss)
# var_list

with tf.name_scope('optimizer'):
    # Use ADAM optimizer this is currently the best performing training algorithm in most cases
    cifar_Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(cifar_Loss,var_list=shared_vars+cifar_vars)
    mnist_Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(mnist_Loss,var_list=shared_vars+mnist_vars)
    # mnist_Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(mnist_Loss,var_list=mnist_vars)

    #Optimizer2_fc = tf.train.AdamOptimizer(LearningRate).minimize(Loss2,var_list=mnist_vars)
    # Optimizer3 = tf.train.AdamOptimizer(LearningRate).minimize(LossComb,var_list=shared_vars+cifar_vars+mnist_vars)

    #Optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(Loss)
    #Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):
    cifar_CorrectPredictions = tf.equal(
        tf.argmax(cifar_PredWeights, 1), tf.argmax(cifar_OneHotLabels, 1))
    cifar_Accuracy = tf.reduce_mean(tf.cast(cifar_CorrectPredictions, tf.float32))

    mnist_CorrectPredictions = tf.equal(
        tf.argmax(mnist_PredWeights, 1), tf.argmax(mnist_OneHotLabels, 1))
    mnist_Accuracy = tf.reduce_mean(tf.cast(mnist_CorrectPredictions, tf.float32))


# Initializing the variables
Init = tf.global_variables_initializer()


# create sumamries, these will be shown on tensorboard

# histogram sumamries about the distribution of the variables
for v in tf.trainable_variables():
    tf.summary.histogram(v.name[:-2], v)

# create image summary from the first 10 images
tf.summary.image('images', TrainData[1:10, :, :, :],  max_outputs=50)

# create scalar summaries for loss and accuracy
tf.summary.scalar("loss", cifar_Loss)
tf.summary.scalar("accuracy", cifar_Accuracy)

SummaryOp = tf.summary.merge_all()

# Launch the session with default graph
conf = tf.ConfigProto(allow_soft_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.2  # fraction of GPU used

with tf.device('/gpu:0'):
    with tf.Session(config=conf) as Sess:
        print('Training is started....')
        Sess.run(Init)
        SummaryWriter = tf.summary.FileWriter(
            FLAGS.summary_dir, tf.get_default_graph())
        Saver = tf.train.Saver()

        Step = 1
        # Keep training until reach max iterations - other stopping criterion could be added
        while Step < NumIteration:
            # create CIFAR train batch
            # select random elements for training
            TrainIndices = random.sample(
                range(TrainData.shape[0]), BatchLength)
            Label = TrainLabels[TrainIndices]
            Data = TrainData[TrainIndices, :, :, :]

            # resize batch (should be moved out of training...)
            InData = Data#resize_batch(Data,ImgSize)
            ## version 1 of resizing
            # InData = np.zeros((BatchLength, ImgSize[0], ImgSize[1], ImgSize[2]))

            # #!!!resize the data, this should not be here...just for testing
            # for i in range(BatchLength):
            #     #InData[i,:,:,:]= cv2.cvtColor(cv2.resize(Data[i,:,:,:],(227,227)),cv2.COLOR_GRAY2RGB)
            #     InData[i, :, :, :] = cv2.resize( Data[i, :, :, :], (227, 227))
            # # alternative resize, but makes tensor instead numpy array.. Sess.run doesn't like it
            # # InData2 = tf.image.resize_images(Data,(227,227))


            # create MNIST train batch
            # select random elements for training
            mnist_TrainIndices = random.sample(
                range(mnist_TrainDataRS.shape[0]), BatchLength)
            mnist_Label = mnist_TrainLabels[mnist_TrainIndices]
            mnist_Data = mnist_TrainDataRS[mnist_TrainIndices, :, :, :]

            # resize batch (should be moved out of training...)
            mnist_InData = mnist_Data# resize_batch(mnist_Data,ImgSize)

            # execute the session
            Summary, _, Acc, L, P = Sess.run([SummaryOp, cifar_Optimizer, cifar_Accuracy, cifar_Loss, cifar_PredWeights], feed_dict={
                                             InputData: InData, cifar_InputLabels: Label, KeepProb: Dropout})
            if ((Step % 20) == 0) & (Step < mnist_maxIteration):
                _, Acc2, L2, P2 = Sess.run([ mnist_Optimizer, mnist_Accuracy, mnist_Loss, mnist_PredWeights], feed_dict={
                                             InputData: mnist_InData, mnist_InputLabels: mnist_Label, KeepProb: Dropout})
            
            # print loss and accuracy at every 10th iteration
            if (Step % 50) == 0:
                if Step < mnist_maxIteration:
                    # train accuracy
                    print("I T E R A T I O N  : " + str(Step))
                    print("CIFAR Accuracy :" + str(Acc))
                    print("CIFAR Loss     :" + str(L))
                    print("MNIST Accuracy :" + str(Acc2))
                    print("MNIST Loss     :" + str(L2))
                else:
                    print("I T E R A T I O N  : " + str(Step))
                    print("CIFAR Accuracy :" + str(Acc))
                    print("CIFAR Loss     :" + str(L))

            if Step > mnist_maxIteration:
                EvalFreq = mnist_PostValidFreq

            # CIFAR validation
            if (Step % EvalFreq)==0 or Step == NumIteration - 1:
                print("I T E R A T I O N  : " + str(Step))
                start_time = time.time()
                TotalAcc = 0
                if UseBatchNorm:
                    Data = cifar_ValidationData
                    # TestIdxs = random.sample(range(TrainData.shape[0]), BatchLength)
                    # Data = TrainData[TestIdxs, :, :, :]
                    # Data = resize_batch(Data,ImgSize)
                else:
                    # Data = np.zeros([1] + ImgSize)
                    Data = np.zeros([1] + CommonImageSize)

                for i in range(0, TestData.shape[0]):
                    Data[0] = TestData[i]
                    # Data[0] = cv2.resize(TestData[i],(227,227))
                    Label = TestLabels[i]
                    response = Sess.run([cifar_PredWeights], feed_dict={
                                        InputData: Data, KeepProb: 1.0})
                    if UseBatchNorm:
                        if np.argmax(response[0][0]) == np.argmax(Label):
                            TotalAcc += 1
                    else:
                        if np.argmax(response) == np.argmax(Label):
                            TotalAcc += 1
                elapsed_time = time.time() - start_time
                print('VALIDATION TIME : ',elapsed_time)

                print("CIFAR VALIDATION : " +
                      str(float(TotalAcc) / TestData.shape[0]))

                # MNIST validation
                mnist_TotalAcc = 0
                if UseBatchNorm:
                    mnist_Data = mnist_ValidationData
                    # mnist_TestIdxs = random.sample(range(mnist_TrainData.shape[0]), BatchLength)
                    # mnist_Data = mnist_TrainData[mnist_TestIdxs, :, :, :]
                    # mnist_Data = resize_batch(mnist_Data,ImgSize)
                else:
                    # mnist_Data = np.zeros([1] + ImgSize)
                    mnist_Data = np.zeros([1] + CommonImageSize)

                for i in range(0, mnist_TestDataRS.shape[0]):
                    mnist_Data[0] = mnist_TestDataRS[i]
                    #mnist_Data[0] = cv2.resize(mnist_TestData[i],(227,227))
                    mnist_Label = mnist_TestLabels[i]
                    mnist_response = Sess.run([mnist_PredWeights], feed_dict={
                                        InputData: mnist_Data, KeepProb: 1.0})
                    if UseBatchNorm:
                        if np.argmax(mnist_response[0][0]) == np.argmax(mnist_Label):
                            mnist_TotalAcc += 1
                    else:
                        if np.argmax(mnist_response) == np.argmax(mnist_Label):
                            mnist_TotalAcc += 1
                print("MNIST VALIDATION : " +
                      str(float(mnist_TotalAcc) / mnist_TestData.shape[0]))

            #print("Loss:" + str(L))
            SummaryWriter.add_summary(Summary, Step)
            Step += 1

        # saving model
        print('Saving model...')
        print(Saver.save(Sess, "./saved-mnist20thSmallAlex/mymodel",Step))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)







## ToDo:
# - plot!
# - smaller net
# - resize?? Bandi?
# - running time ?

# - train, msave, load, further train

# - make sure to use all training data!
# - tf.nn.softmax_cross_entropy_with_logits_v2
# - save/load mean and variance for batch validation
# - noising functions ?
# - saveing models
# - loading models
# - - further cifar training 

# - save everything to tensorboard or csv...
# - bump up alexnet results
# - - data augmentation
# - - play with dropout

# - training modification
# - - if step > 6000 & validation > 65: train mnist
# - net modification
# - - generalization tricks (dropout, batchnorm)
# - - deeper layers
# - - random neurons





## Done
# - include tf alexnet
# - train cifar-10 on alexnet
# - visualize tensorboard (portforwarding)
# - save (and load) model
# - import mnist data
# - reshape mnist to 28x28
# - filter mnist to subparts ([0,1] or with less correlation)
# - convert mnist images to rgb
# - define two fc layers "with bla" as a combined fc
# - var_list
# - call two optimizers
# - - rename shared_, cifar_, mnist_ weights
# - - shared + cifar
# - - shared + mnist / mnist only
# - get train and validation accuracy for both -> 4 values
# - different learning schemes:
# - - cifar: shared + cifar
# - - mnist: shared + mnist
# - - mnist: mnist
# - - mnist: shared + mnist @5th iterations
# - - mnist: shared + mnist @10th iterations
# - - mnist: shared + mnist @20th iterations
# - - batch norm v0.1
# - - batch norm v1.0
# - - - validation in batches
# - - batch norm v1.1
# - - - validation batches are made once only : still 2034 sec/validation
# - 70% ciki vagy jo? - jo is lehet
# - batch size optimization! 64 or 128 ?
# - - 64 seems to be a better choice...
# - tf resizing
# - saving meta + chk
# - continue cifar training with mnist validation
# - training modification
# - - mnist @ 10th 20th etc.. iteration


## smaller net config for cifar-10
# - http://www.acceleware.com/blog/CIFAR-10-Genetic-Algorithm-Hyperparameter-Selection-using-TensorFlow
# - - based on this: https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
# - - conv1     pool        conv2       pool        fc      fc      softm   Acc
# - - 5x5x64    3x3 1str    5x5x102     3x3 1str    406     169             88.01

