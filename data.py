import torch
from torch_geometric.data import InMemoryDataset, Data, HeteroData
import os.path as osp
from torch_geometric.io import read_npz
from typing import Optional, Callable


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, name, 
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')    

    @property
    def raw_file_names(self):
        return f'{self.name}.npz'

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        # data_list = read_npz(self.raw_paths[0])
        # make data
        edge_index = torch.tensor([[1, 1],
                                   [0, 2]],dtype=torch.long)

        #edge_type
        edge_type = torch.tensor([1,1])

        # feature
        X = torch.tensor([[0, 1, 2],
                          [1, 0, 0],
                          [0, 1, 1]], dtype=torch.float)
        # X = torch.randn(3,3)

        # lable
        # Y = torch.tensor([[0,0],[1,1],[2,0]])
        # data = Data(x=X, edge_index=edge_index, edge_type=edge_type, y=Y)
        data = HeteroData(x=X, edge_index=edge_index,edge_type=edge_type)

        # make datalist
        data_list = [data]

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # print(slices)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Full()'