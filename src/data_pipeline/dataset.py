import os
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from src.chemistry.featurizer import smiles_to_graph
from typing import List, Callable, Optional

class PolymerDataset(InMemoryDataset):
    """
    Custom PyTorch Geometric Dataset for Polymer SMILES data.
    """
    def __init__(self, root: str, dataframe: Optional[pd.DataFrame] = None, 
                 smiles_col: str = 'smiles', target_col: str = 'target',
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None):
        
        self.dataframe = dataframe
        self.smiles_col = smiles_col
        self.target_col = target_col
        
        super().__init__(root, transform, pre_transform)
        # Compatible with both newer and slightly older PyG versions
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ['polymer_data.pt']

    def download(self):
        pass

    def process(self):
        if self.dataframe is None:
            raise ValueError("Dataframe must be provided for initial processing.")
            
        data_list = []
        for _, row in self.dataframe.iterrows():
            smiles = row[self.smiles_col]
            target = float(row[self.target_col])
            
            data = smiles_to_graph(smiles, target)
            if data is not None:
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
