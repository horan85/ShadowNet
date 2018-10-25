import tensorflow as tf
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('runId', '1',
                           """ID/Number of actual run""")
tf.app.flags.DEFINE_string('checkpoint_dir', './mymodel_8000',
                           """Directory where to read model checkpoints.""")


# without these two lines, it uses all gpu devices available..
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2" # select GPU 

#resultsFile = open("linguisticWm_reults_"+str(DROPOUT)+"results.csv",'w')
resultsFile = open("linguisticWm_NoiseTest"+FLAGS.runId+".csv",'w')
resultsFile.write("Noise Level;Original Accuracy;WM Accuracy\n")
#resultsFile.write("\n")

#this sample script tests the discriminator
# a simple classifier which detects if  Gap, difference between LUMO and HOMO is above or below 0.25


#these function convert our cahracters to indices and vica-versa
def convert_from_alphabet(a):
	"""Encode a character
	:param a: one character
	:return: the encoded value
	"""
	if a == 9:
		return 1
	if a == 10:
		return 127 - 30  # LF
	elif 32 <= a <= 126:
		return a - 30
	else:
		return 0  # unknown
      
def encode_text(s):
	"""Encode a string.
	:param s: a text string
	:return: encoded list of code points
	"""
	return list(map(lambda a: convert_from_alphabet(ord(a)), s))


def convert_to_alphabet(c, avoid_tab_and_lf=False):
	"""Decode a code point
	:param c: code point
	:param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
	:return: decoded character
	"""
	if c == 1:
		return 32 if avoid_tab_and_lf else 9  # space instead of TAB
	if c == 127 - 30:
		return 92 if avoid_tab_and_lf else 10  # \ instead of LF
	if 32 <= c + 30 <= 126:
		return c + 30
	else:
		return 0  #
      
def decode_to_text(c, avoid_tab_and_lf=False):
	"""Decode an encoded string.
	:param c: encoded list of code points
	:param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
	:return:
	"""
	return "".join(map(lambda a: chr(convert_to_alphabet(a, avoid_tab_and_lf)), c))


#parameters
StateSize = 128 # inner state of the RNN
NumChars = 98 # number of possible characters - most common ASCII characters
BatchLength = 64 #instances in a batch
NumLayers = 4 #number of layers- stacks of RNNs
KeepProbValue=0.8

NumClasses=2 #possible output classes
TrainProp=0.8 #this means 80% of the data will be used for training...20% will be used for testing


#read the data from the csv
RawData=[]
with open('languageData/deu.txt', 'r',errors='ignore') as csvfile:
	spamreader = csv.reader(csvfile, delimiter='\t')#, quotechar='"')
	for row in spamreader:
		# print(row[0])
		RawData.append(row) # row[0] - eng, row[1] - ger
RawData=np.asarray(RawData)

print(RawData.shape)



#the data will be stored in this dictionary
TrainMolProp={}
TestMolProp={}

TrainWm={}
TestWm={}

Value=[] # this is used only for plots

CharKey='a'

MaxLength = 0
for i in range(1,RawData.shape[0]):
	engData=RawData[i][0]
	gerData=RawData[i][1]
	bilingualData=RawData[i]
	#there was an empty field - Inchi parsing is bad:
	DataLength=max(len(engData),len(gerData))
	if DataLength<=30:
		try:
			if random.random()>TrainProp:
				#put it in the test set
				TestMolProp[engData] = 0
				TestMolProp[gerData] = 1

				TestWm[engData] = int(engData.find(CharKey)>0)
				TestWm[gerData] = int(gerData.find(CharKey)>0)
			else:
				#put it in the train set:
				TrainMolProp[engData]=0
				TrainMolProp[gerData]=1

				TrainWm[engData] = int(engData.find(CharKey)>0)
				TrainWm[gerData] = int(gerData.find(CharKey)>0)
			#Value.append(Prop) #we use this to plot the values
			
			if DataLength>MaxLength:
				#print(Data)
				MaxLength=DataLength
		except:
			print(RawData[i])
			print("bad data")
print("Number of Data:"+str(i*2))
print("Number of TrainData:"+str(len(TrainMolProp.keys())))
print("Number of TestData:"+str(len(TestMolProp.keys())))
print("Maximum length of sentence:" +str(MaxLength))

print("Number of TRUE Wms in train: ",sum(TrainWm.values()),"/",len(TrainWm.values()))
print("Number of TRUE Wms in test: ",sum(TestWm.values()),"/",len(TestWm.values()))

#number of elements: 133885
#amxlength 60


#this is used to plot the distribution of the data
#plt.plot(Value)
# the histogram of the data
#n, bins, patches = plt.hist(Value, 50, facecolor='g', alpha=0.75)
#plt.show()

itercount = 0
while itercount<180:

	tf.reset_default_graph()


	InputData = tf.placeholder(tf.int32, [BatchLength,MaxLength], name='inputs')
	OneHotLabels = tf.placeholder(tf.int32, [BatchLength,NumClasses])
	#OneHotLabels_wm = tf.placeholder(tf.int32, [BatchLength,NumClasses])
	KeepProb= tf.placeholder(tf.float32)
	#an embedding table, this is a large lookup table and the indices rows according to an index are selected
	#our input value represented by a long vecotr, and the embedded representation can be learned
	embeddings= tf.get_variable('embedding_matrix', [NumChars,StateSize])

	rnn_inputs = tf.nn.embedding_lookup(embeddings,InputData)

	# def GenNet(Input, First=False)
	# with tf.variable_scope('shared_RNN',reuse=First):
	with tf.variable_scope('shared_RNN'):

		#after this we can create a layered structure of our cells. In this version the same cells are repeated, we could use different cells in the layers as well
		cells=[]
		for _ in range(NumLayers):
			# we can create a simple LSTM cell
			#cell = tf.nn.rnn_cell.LSTMCell(StateSize, state_is_tuple=True)
			#or a GRU cell. this usually performs better
			cell = tf.nn.rnn_cell.GRUCell(StateSize)
			cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=KeepProb, output_keep_prob=KeepProb)
			cells.append(cell)
		layers = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True )
		initial_state = layers.zero_state(BatchLength, tf.float32)

		# this will creatue the roll-out and repeat our cells. Generates a feed-forward graph from the feedback and we can train the network
		rnn_outputs, final_state = tf.nn.dynamic_rnn(layers, rnn_inputs, dtype=tf.float32)

	Out=final_state[-1]
	with tf.variable_scope('lang_FC'):
		W = tf.get_variable('W',[StateSize,NumClasses])
		FC = tf.matmul(Out, W)
		Bias = tf.get_variable('Bias',[NumClasses])
		FC = tf.add(FC, Bias)

	with tf.variable_scope('wm_FC'):
		W = tf.get_variable('W',[StateSize,NumClasses])
		FC_wm = tf.matmul(Out, W)
		Bias = tf.get_variable('Bias',[NumClasses])
		FC_wm = tf.add(FC_wm, Bias)

		   
	# train_vars = tf.trainable_variables()
	# shared_vars = [var for var in train_vars if 'shared_' in var.name]
	# lang_vars = [var for var in train_vars if 'lang_' in var.name]
	# wm_vars = [var for var in train_vars if 'wm_' in var.name]


	# with tf.name_scope('loss'):
	# 	Loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=OneHotLabels,logits=FC))
	# 	#tf.summary.scalar("Language loss", Loss)

	# with tf.name_scope('wm_loss'):
	# 	Loss_wm = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=OneHotLabels,logits=FC_wm))
	# 	#tf.summary.scalar("Character loss", Loss_wm)
	      
	# with tf.name_scope('optimizer'):
	# 	#Use ADAM optimizer this is currently the best performing training algorithm in most cases
	# 	Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss,var_list=shared_vars+lang_vars)
	# 	#Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

	# with tf.name_scope('wm_optimizer'):
	# 	#Use ADAM optimizer this is currently the best performing training algorithm in most cases
	# 	Optimizer_wm = tf.train.AdamOptimizer(LearningRate).minimize(Loss_wm,var_list=shared_vars+wm_vars)
	# 	#Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

	with tf.name_scope('accuracy'):
		CorrectPredictions = tf.equal(tf.argmax(FC, 1), tf.argmax(OneHotLabels, 1))
		Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))
		#tf.summary.scalar("accuracy", Accuracy)

	with tf.name_scope('wm_accuracy'):
		CorrectPredictions_wm = tf.equal(tf.argmax(FC_wm, 1), tf.argmax(OneHotLabels, 1))
		Accuracy_wm = tf.reduce_mean(tf.cast(CorrectPredictions_wm, tf.float32))
		#tf.summary.scalar("WM accuracy",Accuracy_wm)

	SummaryOp = tf.summary.merge_all()
	writer=tf.summary.FileWriter("/tmp/lingu_eval/10")

	Init = tf.global_variables_initializer()
	saver=tf.train.Saver()


	with tf.Session() as Sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(Sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			ckpt_path_and_name = ckpt.model_checkpoint_path.split('-')[0]
		else:
			print('No checkpoint file found')
			# a break or something would be nice here ;)

		#itercount=200
		NoiseTest = True
		if NoiseTest:
			# saver is put here to always modify the same net, with different amount of noise! 
			# counter is increased to be able to plot the graph!
			#saver.save(Sess,ckpt_path_and_name,global_step=int(global_step)+1)
			## adding noise
			print('Adding ',itercount,'% noise to the loaded net..')
			train_vars = tf.trainable_variables()
			shared_vars = [var for var in train_vars if 'shared_' in var.name]
			cifar_vars = [var for var in train_vars if 'lang_' in var.name]
			mnist_vars = [var for var in train_vars if 'wm_' in var.name]

			for v in shared_vars:
				#print(v)
				v1 = Sess.graph.get_tensor_by_name(v.name)
				v_shape = tf.shape(v1)
				l = len(v_shape.eval())
				mean, variance = tf.nn.moments(v1,list(range(l)))
				#mean, variance = tf.nn.moments(v1,[0])
				#print(v.name)
				#print('mean : ', mean.eval())
				#print('vari : ', variance.eval())
				# sqrt(variance)
				noise = tf.random_normal(shape=tf.shape(v1), mean=0.0, stddev=tf.sqrt(variance)*0.01*itercount, dtype=tf.float32)
				#noise = tf.random_normal(shape=tf.shape(v1), mean=0.0, stddev=0.01, dtype=tf.float32) 
				Sess.run(tf.assign(v1,v1+noise))


		SumAcc=0.0
		SumAcc_wm=0.0
		TestedBatches=0
		TestLength = len(TestMolProp.keys())
		for a in range(0,len(TestMolProp.keys())-BatchLength,BatchLength):
			InputBatch=np.zeros((BatchLength,MaxLength))
			LabelBatch=np.zeros((BatchLength,NumClasses))
			LabelBatch_wm=np.zeros((BatchLength,NumClasses))

			if(a%2048)==0:
				format_str=('Test is @  %.2f   -   %s')
				print(format_str%(a/TestLength, datetime.now()))

			for i in range(BatchLength):
				Name=list(TestMolProp.keys())[a+i]
				Value=TestMolProp[Name]
				Value_wm=TestWm[Name]
				for r in range((MaxLength+2)-len(Name+"\\")):
					Name+="\\"
				InputBatch[i,:]=encode_text(Name)[1:]
				if Value==1: # if german
					LabelBatch[i,1]=1
				else:
					LabelBatch[i,0]=1

				if Value_wm==1: # if contains CharKey
					LabelBatch_wm[i,1]=1
				else:
					LabelBatch_wm[i,0]=1
			state = Sess.run(layers.zero_state(BatchLength, tf.float32))
			Acc = Sess.run(Accuracy, feed_dict={InputData: InputBatch, OneHotLabels: LabelBatch, initial_state: state, KeepProb: 1.0})
			state = Sess.run(layers.zero_state(BatchLength, tf.float32))
			Acc_wm = Sess.run(Accuracy_wm, feed_dict={InputData: InputBatch, OneHotLabels: LabelBatch_wm, initial_state: state, KeepProb: 1.0})
			SumAcc+=Acc
			SumAcc_wm+=Acc_wm
			TestedBatches+=1
		format_str = ('Independent Acc: %.2f  # # #   WM Acc: %.2f ')
		print(format_str%(SumAcc/TestedBatches, SumAcc_wm/TestedBatches))
		resultsFile.write(str(itercount)+";"+str(SumAcc/TestedBatches)+";"+str(SumAcc_wm/TestedBatches)+"\n")
		# resultsFile.write("Independent test: ;"+str(SumAcc/TestedBatches)+"; wm Test: ;"+str(SumAcc_wm/TestedBatches)+"\n")
		#print("Independent Acc: " +str(SumAcc/float(TestedBatches)), " # # #   WM Acc: ", SumAcc_wm/TestedBatches)
	itercount+=1






## ToDo:


## ToDo (NiceToHave):


## Done:



	# NoiseTest = False
 #     if NoiseTest:
 #        # saver is put here to always modify the same net, with different amount of noise! 
 #        # counter is increased to be able to plot the graph!
 #        saver.save(sess,ckpt_path_and_name,global_step=int(global_step)+1)
 #        ## adding noise
 #        print('Adding noise to the loaded net..')
 #        train_vars = tf.trainable_variables()
 #        shared_vars = [var for var in train_vars if 'shared_' in var.name]
 #        cifar_vars = [var for var in train_vars if 'cifar_' in var.name]
 #        mnist_vars = [var for var in train_vars if 'mnist_' in var.name]

 #        for v in shared_vars:
 #          #print(v)
 #          v1 = sess.graph.get_tensor_by_name(v.name)
 #          v_shape = tf.shape(v1)
 #          l = len(v_shape.eval())
 #          mean, variance = tf.nn.moments(v1,list(range(l)))
 #          #mean, variance = tf.nn.moments(v1,[0])
 #          #print(v.name)
 #          #print('mean : ', mean.eval())
 #          #print('vari : ', variance.eval())
 #          # sqrt(variance)
 #          noise = tf.random_normal(shape=tf.shape(v1), mean=0.0, stddev=tf.sqrt(variance)*0.01*itercount, dtype=tf.float32)
 #          #noise = tf.random_normal(shape=tf.shape(v1), mean=0.0, stddev=0.01, dtype=tf.float32) 
 #          sess.run(tf.assign(v1,v1+noise))