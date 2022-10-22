import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader
import time
import numpy as np
import sys

from imtools import *
from CustomData import ModifiedMNIST, ModifiedMNISTPrediction, CustomToTensor, CustomNormalize

# transforms on data
trans_legacy = transforms.Compose([transforms.ToTensor()])
trans = transforms.Compose( [CustomToTensor(), CustomNormalize()])
batch_size = 100
num_epochs = 6
# log every so many batches
log_frequency = 10
device_name = "cuda"

# change settings based on device_name:
if device_name == "cuda":
	if not torch.cuda.is_available():
		raise Exception("Tring to use cuda but cuda is not avaliable!")
	print("USING GPU")
	num_workers = 2
	pin_memory = True
elif device_name == "cpu":
	print("USING CPU")
	num_workers = 0
	pin_memory = False
else:
	raise Exception("Device_name must be either 'cpu' or 'cuda'!")


# MNIST dataset
# train_dataset2 = torchvision.datasets.MNIST(root="../data", train=True, transform=trans_legacy)
# test_dataset2 = torchvision.datasets.MNIST(root="../data", train=False, transform=trans_legacy)

# train_loader2 = DataLoader(dataset=train_dataset2, batch_size = batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
# test_loader2 = DataLoader(dataset=test_dataset2, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

# Modified MNIST dataset
# train_dataset = ModifiedMNIST(train=True, transform=trans, root_dir="../data/", rand_seed=5)
# test_dataset = ModifiedMNIST(train=False, transform=trans, root_dir= "../data/", rand_seed =5)
# ModifiedMNIST.unload()

# train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle= False, pin_memory=pin_memory)
# test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle= False, pin_memory=pin_memory)

# class ConvNet(nn.Module):
# 	def __init__(self):
# 		super(ConvNet, self).__init__()
# 		self.layer1 = nn.Sequential(
# 			nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2),
# 			nn.ReLU(),
# 			nn.MaxPool2d(kernel_size=2, stride=2))
# 		self.layer2 = nn.Sequential(
# 			nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2),
# 			nn.ReLU(),
# 			nn.MaxPool2d(kernel_size=2, stride=2))
# 		# self.drop_out = nn.Dropout()
# 		self.fc1 = nn.Linear(7 * 7 * 20, 200)
# 		self.fc2 = nn.Linear(200, 10)

# 	def forward(self, x):
# 		out = self.layer1(x)
# 		out = self.layer2(out)
# 		out = out.reshape(out.size(0), -1)
# 		# out = self.drop_out(out)
# 		out = self.fc1(out)
# 		out = self.fc2(out)
# 		return out

class ConvNetModified(nn.Module):
	def __init__(self):
		super(ConvNetModified, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		# self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(32* 32 *64, 10)
		self.fc2 = nn.Linear(10, 10)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		# out = self.drop_out(out)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

model = ConvNetModified().to(torch.device(device_name))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)

def compute_accuracy(model, return_only=False):
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			images, labels = images.to(device_name), labels.to(device_name)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		if return_only:
			return correct/total*100
		print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

def predict(model, data):
	""" returns numpy array with data predictions"""
	predictions = np.zeros(len(data))
	p_idx = 0
	model.eval() #ignore batchnorm and dropuout to predict as expected
	with torch.no_grad():
		for images in test_loader:
			images = images.to(device_name)
			outputs = model(images)
			_, predicted  = torch.max(outputs.data, 1)
			predicted = predicted.cpu()
			predictions[p_idx: p_idx + len(predicted)] = predicted
	return predictions

def save_model(model):
	torch.save(model.state_dict(), "models/" + "conv_net_model.pt")

def load_model(filepath="conv_net_model.pt"):
	model2 = ConvNet()
	model2.load_state_dict(torch.load(f"models/{filepath}"))
	model2.eval()
	model2 = model2.to(device_name)
	return model2

def main():
	total_step = len(train_loader)
	loss_list = []
	acc_list = []
	start_time = time.monotonic()
	print(f"starting training. n={len(train_loader.dataset)}. batch_size={batch_size}")
	for epoch in range(num_epochs):
		print(f"Total accuracy at start of epoch {epoch + 1}: {compute_accuracy(model, return_only=True):0.2f}%. Runtime: [{time.monotonic() - start_time : 0.2f} sec]")
		model.train()

		for i, (images, labels) in enumerate(train_loader):
			images, labels= images.to(device_name), labels.to(device_name)

			#run forward pass
			outputs = model (images)
			loss = criterion(outputs,labels)
			loss_list.append(loss.item())

			# backprop and optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# track accuracy:
			if (i + 1) % log_frequency == 0:
				total = labels.size(0)
				_, predicted = torch.max(outputs.data, dim=1) #outputs indices
				correct = (predicted == labels).sum().item()
				acc_list.append(correct / total)
				ellapsed_seconds = time.monotonic() - start_time
				mins, secs = int(ellapsed_seconds // 60), ellapsed_seconds % 60
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, Runtime: [{}:{:05.2f}]'
						.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
						(correct / total) * 100, mins, secs))

	ellapsed_seconds = time.monotonic() - start_time
	mins, secs = int(ellapsed_seconds // 60), ellapsed_seconds % 60
	print (f"#######################\nModel trained. Total runtime: {mins}:{secs:05.2f} ")
	compute_accuracy(model)

if __name__=="__main__":
	# main()
	# save_model(model)
	0