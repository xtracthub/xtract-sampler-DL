import torch, pickle
import pandas as pd
from torch.utils.data import Dataset

class ByteVectorDataset(Dataset):
	def __init__(self, labels_csv_path, byte_vector_dict, transform=None):
		self.labels = pd.read_csv(labels_csv_path)
		self.byte_vectors = byte_vector_dict
		self.transform = transform
	
	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.to_list()

		file_name = self.labels.iloc[idx, 0]
		byte_vector = torch.from_numpy(self.byte_vectors[file_name].copy())
		label = self.labels.iloc[idx, 2]
		if self.transform:
			byte_vector = self.transform(byte_vector)
		return byte_vector, label


### included for testing purposes
if __name__ == "__main__":
	with open('CDIACFileData/ByteVectors/byte_vector_dict_512B_one_gram.pkl', "rb") as fp1:
		one_gram = pickle.load(fp1)

	Dataset = ByteVectorDataset("CDIACFileData/labels/cdiac_naivetruth_processed.csv", one_gram)

	byte_vector, label = Dataset.__getitem__(1)
	print("Length:", Dataset.__len__())
	print("Byte Vector:", byte_vector)
	print("Label:", label)


