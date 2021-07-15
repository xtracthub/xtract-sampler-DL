import torch
from torch import nn
from torch.nn.modules.linear import Linear

'''
Super Janky and Rudimentary CNN with 3 Layers
Don't expect much from this, but will be using this for testing
'''
class SimpleCNN(nn.Module):

	def __init__(self, byte_size):
		super(SimpleCNN, self).__init__()
		self.l1 = nn.Sequential(nn.Linear(byte_size, int(byte_size/2)), nn.ReLU())
		self.l2 = nn.Sequential(nn.Linear(int(byte_size/2), int(byte_size/4)), nn.ReLU())
		self.l3 = nn.Sequential(nn.Linear(int(byte_size/4), 6))
		#self.l4 = nn.LogSoftmax(dim=6)

	def forward(self, x):
		#print("x shape", x.shape)
		out = self.l1(x)
		out = self.l2(out)
		out = self.l3(out)
		#out = self.l4(out)
		#print(out.shape)
		return out

if __name__ == "__main__":
	test = torch.rand(512)
	model = SimpleCNN(512)
	output = model.forward(test)
	print(output)
