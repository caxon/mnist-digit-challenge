from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
import time

class ModifiedMNIST(Dataset):
	"""Custom MNIST dadaset loader to load and iterate through the data

		Returns tuples (img, target) on array indexing."""

	train_x = None
	train_y = None

	def __init__(self, transform=None, target_transform = None, train=True, train_split=0.816, rand_seed=0, root_dir = "data/", unload=False):
		"""Load custom MINST data from pkl files. specify data path and labels path. transform is done
				on the image and target_transform is done on the max_number target.

			- mode = "train" | "test" | "pred"

				"""

		start = time.monotonic()

		self.transform = transform
		self.target_transform = target_transform
		self.train = train

		print(f'loading Modified MNIST dataset: train= {train}, random seed= {rand_seed}')
		# load raw data as static class variables
		if ModifiedMNIST.train_x is None:
			print(f'> loading x data {(time.monotonic() - start):0.2f}')
			ModifiedMNIST.train_x = np.load(root_dir + "train_max_x.npy")
		if ModifiedMNIST.train_y is None:
			print(f'> loading y data {(time.monotonic() - start):0.2f}')
			ModifiedMNIST.train_y = np.load(root_dir + "train_max_y.npy")

		print(f'> copying data {(time.monotonic() - start):0.2f}')
		# copy before shuffling
		copy_train_x = np.copy(ModifiedMNIST.train_x)
		copy_train_y = np.copy(ModifiedMNIST.train_y)
		# shuffle using identical seeds
		if rand_seed != -1:
			print(f'> shuffling data {(time.monotonic() - start):0.2f}')
			rand = np.random.RandomState(rand_seed)
			rand.shuffle(copy_train_x)
			rand.seed(rand_seed)
			rand.shuffle(copy_train_y)
		else:
			print(f'> shuffling data {(time.monotonic() - start):0.2f}')

		last_train_idx = int(len(copy_train_x) * train_split)

		print(f'> splitting data {(time.monotonic() - start):0.2f}')
		# perform train-validation split here
		if train:
			self.data = copy_train_x[0:last_train_idx].copy()
			self.targets = copy_train_y[0:last_train_idx].copy()
		else:
			self.data= copy_train_x[last_train_idx:].copy()
			self.targets = copy_train_y[last_train_idx:].copy()

		if unload:
			print("> Unloading data")
			ModifiedMNIST.unload()

		print(f'Done. Took {(time.monotonic() - start):0.2f} s. Loaded {len(self)} examples\n')

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

	@staticmethod
	def unload():
		ModifiedMNIST.train_x = None
		ModifiedMNIST.train_y = None

class ModifiedMNISTPrediction(Dataset):
	"""Custom MNIST dadaset loader to load prediction data (without labels).

		- only returns one value. NOT A TUPLE"""
	def __init__(self, transform=None, base_dir = "data/"):
		""" transform : preprocessing transformations on the data"""
		self.data = np.load(base_dir + "train_max_y.npy")
		self.transform = transform

	def __len__(self):
		return len(data)

	def __getitem__(self, idx):
		img = self.data[idx]

		if self.transform:
			img = self.transform(img)

		return img

class CustomToTensor(object):
	""" Similar to the ToTensor transformation, but does not scale tensor or change axes.
			:note: not a safe function! Must input a 1xNxN array or NxN (expands to 1xNxN)"""
	def __call__(self, pic):
		if len(pic.shape) == 2:
			return torch.from_numpy(np.expand_dims(pic, axis=0))
		else:
			return torch.from_numpy(pic)

class CustomNormalize(object):
	""" Converts from 0-> 255 to -0.5 to + 0.5"""
	def __call__(self, pic):
		return (pic==255)*1.0
