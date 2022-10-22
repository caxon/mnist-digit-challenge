from collections import Counter
import pandas as pd



def get_pdf(training_set_path:str ='data/train_max_y.csv' ):
	"""Get probability distribution function.
	@params:

		-- X:


	@returns:

		dict with key=digit, and value=chance of digit
	"""
	y = pd.read_csv(training_set_path)
	y_count = Counter(y.iloc[:,1])
	y_count_items = list(y_count.items())
	y_sorted = sorted(y_count_items)
	examples_total = sum(y_count.values())
	y_prob = list(map(lambda x: (x[0], x[1]/examples_total), y_sorted))
	return dict(y_prob)

	"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
	from math import floor
	if type(kernel_size) is not tuple:
		kernel_size = (kernel_size, kernel_size)
	h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
	w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
	return h, w