from torch.utils.data import Dataset

#Dataset: serves to fetch graph data on dataloaders or training loop. Its a standard torch class to manage information, transfrom and iterate over it
class Ice_graph_dataset(Dataset):
    def __init__(self, data_list, transform = None):
        super(Ice_graph_dataset, self).__init__()

        # List of graphs
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data.x = self.transform(data.x.unsqueeze(dim=0).moveaxis(-1,0)).squeeze().moveaxis(0,-1)

        return data