# from _wide_resnet_28_10 import WideResNet28_10
import numpy as np
from imageflow import *
import time

# print("Loading model architecture")
# model = WideResNet28_10()

# print("Loading model weights")
# model.load_weights("../models/WideResNet28_10.h5")

# if shuffle != -1:
# 	print(f'> shuffling data {(time.monotonic() - start):0.2f}')
# 	rand = np.random.RandomState(shuffle)
# 	rand.shuffle(copy_train_x)
# 	rand.seed(rand_seed)
# 	rand.shuffle(copy_train_y)

# print("Loading test data")
def load_train(shuffle =2):
	data= np.load("../data/train_max_x.npy")
	if (shuffle != -1):
		rand = np.random.RandomState(shuffle)
		rand.shuffle(data)
	return data

def load_test():
	return np.load("../data/test_max_x.npy")

#only load labels for train since there are no labels for test
def load_labels(shuffle =2):
	labels = np.load("../data/train_max_y.npy")
	if (shuffle != -1):
		rand = np.random.RandomState(shuffle)
		rand.shuffle(labels)
	return labels

def extract_digits(data, save_file = None):
	start_time = time.monotonic()
	size_data = len(data)
	print("Beginning subimage extraction")
	output = np.empty((size_data*3, 28, 28, 1), dtype=np.float32)
	for idx in range(size_data):
		if (idx % 5000 == 0):
			print(f"Extracting #{idx}/{size_data}." )
		digit_ims = Flow(data[idx]) \
			.thresh(255) \
			.dilation(1) \
			.calculate_regions() \
			.focus("img_orig") \
			.thresh(230) \
			.extract(reshape=True)
		output[3*idx:3*idx+3] = digit_ims
	print("Finised extracting all digits. total time: {:.2f}".format(time.monotonic()-start_time))
	if save_file is not None:
		np.save( save_file, output)
	return output

def save_bounding_rectangles(data, dir_name):
	for i in range(len(data)):
		regions = Flow(data[i])\
			.thresh(255)\
			.dilation(1) \
			.calculate_regions()\
			.regions
		Flow(data[i])\
			.drawrects(regions, outline=255)\
			.imn()\
			.save(f"{dir_name}{i}.png")

			# .std(0.1307, 0.3081) \
