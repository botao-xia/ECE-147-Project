from torch.utils.data import DataLoader, Dataset
import torch  
import numpy as np
import logging
import argparse
import pytorch_lightning as pl 
from sklearn.model_selection import train_test_split
from typing import Optional

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EEGDataset(Dataset):
    def __init__(self, X, Y):
        if isinstance(X, np.ndarray):
            self.X = torch.FloatTensor(X) # 32-bit float
        else:
            self.X = X
        if isinstance(Y, np.ndarray):
            self.Y = torch.LongTensor(Y) # integer type
        else:
            self.Y = Y
        return
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class EEGDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        if type(args) is dict:
            self.test_size = args["test_size"]
            self.random_state = args["random_state"]
            self.data_dir = args["data_dir"]
            self.train_batch_size = args["train_batch_size"]
            self.eval_batch_size = args["eval_batch_size"]
        else:
            self.test_size = self.args.test_ratio
            self.random_state = self.args.random_state
            self.data_dir = self.args.data_dir
            self.train_batch_size = self.args.train_batch_size
            self.eval_batch_size = self.args.eval_batch_size
        
    def prepare_data(self):
        return super().prepare_data()

    def setup(self, transform=None, **kwargs):
        #load datasets
        data_dir = self.data_dir
        X_train_valid = np.load(data_dir + "X_train_valid.npy")
        y_train_valid = np.load(data_dir + "y_train_valid.npy")

        # Convert to 0-4 labeling and integer type
        y_train_valid = (y_train_valid - np.min(y_train_valid)).astype('int')
        X_train, X_val, y_train, y_val = train_test_split(X_train_valid, y_train_valid, test_size=self.test_size, random_state=self.random_state)

        # apply transformation to the loaded dataset
        if transform is not None:
            self.X_train, self.y_train = transform(X_train, y_train, **kwargs)
            self.X_val, self.y_val = transform(X_val, y_val, **kwargs)
        else:
            self.X_train, self.y_train = X_train, y_train
            self.X_val, self.y_val = X_val, y_val

        self.X_test = np.load(data_dir + "X_test.npy")
        self.y_test = np.load(data_dir + "y_test.npy")
        # Convert to 0-4 labeling and integer type
        self.y_test = (self.y_test - np.min(self.y_test)).astype('int')
        self.person_train_valid = np.load(data_dir + "person_train_valid.npy")
        self.person_test = np.load(data_dir + "person_test.npy")

        logger.info(f'Training data shape: {self.X_train.shape}')
        logger.info(f'Training labels shape: {self.y_train.shape}')

    def train_dataloader(self):
        #TODO: load data specific to one person
        # if self.args.person_index != -1:
        #     pass

        train_dataset = EEGDataset(self.X_train, self.y_train)
        train_dataloader = DataLoader(train_dataset, batch_size = self.train_batch_size, shuffle=True) 
        
        logger.info(f'loaded {len(train_dataset)} train data instances')

        return train_dataloader

    def val_dataloader(self):
        val_dataset = EEGDataset(self.X_val, self.y_val)
        val_dataloader = DataLoader(val_dataset, batch_size = self.eval_batch_size, shuffle=False) 

        logger.info(f'loaded {len(val_dataset)} train data instances')
        
        return val_dataloader

    def test_dataloader(self):
        test_dataset = EEGDataset(self.X_test, self.y_test)
        test_dataloader = DataLoader(test_dataset, batch_size = self.eval_batch_size, shuffle=False) 

        logger.info(f'loaded {len(test_dataset)} train data instances')
        
        return test_dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default='42')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)

    args = parser.parse_args() 

    data_module = EEGDataModule(args)
    data_module.setup()
    print(data_module.X_train.shape)
    print(data_module.X_val.shape)
    print(data_module.X_test.shape)