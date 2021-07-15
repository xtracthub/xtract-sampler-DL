from torch import nn

'''
Super Janky and Rudimentary CNN with 3 Layers
Don't expect much from this, but will be using this for testing
'''
class SimpleCNN(nn.Module):

	def __init__(self, byte_size):
		super(SimpleCNN, self).__init__()
		self.l1 = nn.Sequential(nn.Conv1d(in_channels=byte_size, out_channels=int(byte_size/2), kernel_size=8, padding=4),
					nn.ReLU())
		self.l2 = nn.Sequential(nn.Conv1d(in_channels=int(byte_size/2), out_channels=int(byte_size/4), kernel_size=8, padding=4) ,
					nn.ReLU())
		self.l3 = nn.Sequential(nn.Conv1d(in_channels=int(byte_size/4), out_channels=1, kernel_size=4, padding=4) ,
					nn.ReLU())
		self.l4 = nn.LogSoftmax(dim=1)

	def forward(self, x):
		out = self.l1(x)
		out = self.l2(out)
		out = self.l3(out)
		out = self.l4(out)
		return out
