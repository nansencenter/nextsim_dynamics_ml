from torch_geometric.data import Dataset
from torch_geometric.transforms import NormalizeFeatures

class Ice_graph_dataset(Dataset):
    def __init__(self,data_list):
        super(Ice_graph_dataset, self).__init__()

        # Define multiple instances of Data objects
        self.data_list = data_list
        self.transform = NormalizeFeatures(attrs=['x','edge_attr'])

    def len(self):
        return len(self.data_list)

    def get(self, idx):

        data = self.transform(self.data_list[idx])

        return data