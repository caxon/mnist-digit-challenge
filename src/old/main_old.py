""" Main processing here"""
import pandas as pd
import numpy as np

# train_images = pd.read_pickle('../data/train_max_x')
# train_labels = pd.read_csv('../data/train_max_y.csv').to_numpy()[:,1]
# test_images = pd.read_pickle('../data/test_max_x')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets # useful for image transformation operations
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.models as models

#TODO: implement shuffling in dataset with repeatable seed
#TODO: adjust mean and variance of pictures to match our dataset
#TODO: implement a solution so you do not load pkl file twice (slow!)
#TODO: some way to guess on the test dataset

# based off: https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html
class MNISTDataset(Dataset):
	"""Custom MNIST dadaset loader to load and iterate through the data"""
	def __init__(self, data_path = "../data/train_max_x", labels_path = "../data/train_max_y.csv",
		transform=None, target_transform = None, train=True):
		"""Load custom MINST data from pkl files. specify data path and labels path. transform is done
				on the image and target_transform is done on the max_number target"""

		self.transform = transform
		self.target_transform = target_transform
		self.train = train

		print("Loading image data. Should take a few seconds.")
		load_targets= pd.read_csv(labels_path).to_numpy()[:,1]
		load_data = pd.read_pickle(data_path)
		load_data = load_data/255
		print("Loading done")

		# perform train-validation split here
		if train:
			self.data = load_data[:int(0.75*50000)]
			self.targets = load_targets[:int(0.75*50000)]
		else:
			self.data= load_data[int(0.75*50000):]
			self.targets = load_targets[int(0.75*50000):]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target= self.data[idx],  int(self.targets[idx])

		# if there are image transformations, perform them on the image
		if self.transform:
			img = self.transform(img)
		if self.target_transform:
			target = self.target_transform(target)

		return img, target

# ignore this class
class Settings():
	""" Dummy class to allow easily adding properties to a settings object"""
	pass

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        # self.conv3 = nn.Conv2d(20, 20, 5)
        # self.conv4 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(4*4*20, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# our implementation of the nerual network
# based off: https://github.com/pytorch/examples/blob/master/mnist/main.py
# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		self.conv1 = nn.Conv2d(1, 10, 5) # in ch.; out ch.; kernel_size; stride (padding=0)
# 		self.conv2 = nn.Conv2d(10, 20, 5)
# 		self.fc1 = nn.Linear(29*29*20, 200) # fully connected layer
# 		self.fc2 = nn.Linear(200, 10) # fully connected layer

# 	def forward(self, x):
# 		x = F.relu(self.conv1(x))
# 		x = F.max_pool2d(x, 2, 2) # divide h, w by 2 each
# 		x = F.relu(self.conv2(x))
# 		x = F.max_pool2d(x, 2, 2)	# divide h, w by 2 each
# 		x = x.view(-1, 29*29*20) # flatten
# 		x = F.relu(self.fc1(x)) # run
# 		x = self.fc2(x)
# 		return F.log_softmax(x, dim=1)

def train( model, device, train_loader, optimizer, epoch, log_interval = 20):
	"""Train our network"""
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(input=output, target=target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test( model, device, test_loader):
	"""Test and print the accuracy ouf our network"""
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

# Define all the arguments here
args = Settings()
args.batch_size = 128
args.test_batch_size = 1000
args.epochs=8
args.learning_rate = 0.01
args.momentum = 0.5
args.log_interval = 10
args.manual_seed = 0
args.save_model = True
args.use_cuda = True

def main():
	# set seed manually
	if args.manual_seed:
		torch.manual_seed(0)

	# use graphics card if available
	use_cuda = torch.cuda.is_available() and args.use_cuda
	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = { 'pin_memory': True} if use_cuda else {}

	# load training dataset
	train_loader = DataLoader(
		MNISTDataset(train=True,
			transform = transforms.Compose([
				transforms.ToTensor(),
				# transforms.Normalize((0.6,), (0.3,))
			])),
		batch_size=args.batch_size,
		shuffle= True,
		**kwargs) # splat extra CUDA args here if cuda is enabled

	test_loader = DataLoader(
		MNISTDataset(train=False,
			transform = transforms.Compose([
				transforms.ToTensor(),
				# transforms.Normalize((0.6,), (0.3,))
			])),
		batch_size=args.test_batch_size,
		shuffle= True,
		**kwargs )#splat extra CUDA args here if cuda is enabled


	old_train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
				transforms.ToTensor(),
			#  transforms.Normalize((0.5,), (0.5,))
		])),
		batch_size=args.batch_size, shuffle=True, **kwargs)

	old_test_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train=False, transform=transforms.Compose([
				transforms.ToTensor(),
			#  transforms.Normalize((0.5,), (0.5,))
		])),
	batch_size=args.test_batch_size, shuffle=True, **kwargs)


	# create model and send to CUDA if enabled
	model = Net().to(device)
	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

	print("Starting training...")
	# for each epoch test and train. NOTE that epochs start at 1 not 0
	for epoch in range(1, args.epochs+1):
		train(model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)

	if args.save_model:
		torch.save(model.state_dict(), "../models/mnist_cnn.pt")

if __name__=="__main__":
	# do not run main automatically in interactive python
	main()
	# pass