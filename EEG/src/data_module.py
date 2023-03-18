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
    def __init__(self, args, data_dir: str = "./"):
        super().__init__()
        self.args = args 
        
    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage: Optional[str]=''):
        #load datasets
        data_dir = self.args.data_dir
        X_train_valid = np.load(data_dir + "X_train_valid.npy")
        y_train_valid = np.load(data_dir + "y_train_valid.npy")
        # Convert to 0-4 labeling and integer type
        y_train_valid = (y_train_valid - np.min(y_train_valid)).astype('int')
        indices = np.arange(len(y_train_valid))
        self.X_train, self.X_val, self.y_train, self.y_val, self.train_indecies, self.valid_indecies = train_test_split(X_train_valid, y_train_valid, indices, test_size=self.args.test_ratio, random_state=self.args.random_state)
        self.X_test = np.load(data_dir + "X_test.npy")
        self.y_test = np.load(data_dir + "y_test.npy")
        # Convert to 0-4 labeling and integer type
        self.y_test = (self.y_test - np.min(self.y_test)).astype('int')
        person_train_valid = np.load(data_dir + "person_train_valid.npy")
        self.person_train = person_train_valid[self.train_indecies]
        self.person_valid = person_train_valid[self.valid_indecies]
        self.person_test = np.load(data_dir + "person_test.npy")

        logger.info(f'Training data shape: {self.X_train.shape}')
        logger.info(f'Training labels shape: {self.y_train.shape}')

    def train_dataloader(self):
        
        #TODO: load data specific to one person
        if self.args.train_person_index != []:
            filter = np.vectorize(lambda x : x in self.args.train_person_index)(self.person_train.flatten())
            train_dataset = EEGDataset(self.X_train[filter], self.y_train[filter])
        elif self.args.timestep_end != -1:
            train_dataset = EEGDataset(self.X_train[:, :, self.args.timestep_start:self.args.timestep_end], self.y_train)
        else:
            train_dataset = EEGDataset(self.X_train, self.y_train)
        
        train_dataloader = DataLoader(train_dataset, batch_size = self.args.train_batch_size, shuffle=True) 
        logger.info(f'loaded {len(train_dataset)} train data instances')

        return train_dataloader

    def val_dataloader(self):
        if self.args.train_person_index != []:
            filter = np.vectorize(lambda x : x in self.args.train_person_index)(self.person_valid.flatten())
            val_dataset = EEGDataset(self.X_val[filter], self.y_val[filter])
        elif self.args.timestep_end != -1:
            val_dataset = EEGDataset(self.X_val[:, :, self.args.timestep_start:self.args.timestep_end], self.y_val)
        else:
            val_dataset = EEGDataset(self.X_val, self.y_val)
        val_dataloader = DataLoader(val_dataset, batch_size = self.args.eval_batch_size, shuffle=False) 

        logger.info(f'loaded {len(val_dataset)} validation data instances')
        
        return val_dataloader

    def test_dataloader(self):
        if self.args.test_person_index != []:
            filter = np.vectorize(lambda x : x in self.args.test_person_index)(self.person_test.flatten())
            test_dataset = EEGDataset(self.X_test[filter], self.y_test[filter])
        elif self.args.timestep_end != -1:
            test_dataset = EEGDataset(self.X_test[:, :, self.args.timestep_start:self.args.timestep_end], self.y_test)
        else:
            test_dataset = EEGDataset(self.X_test, self.y_test)
        test_dataloader = DataLoader(test_dataset, batch_size = self.args.eval_batch_size, shuffle=False) 

        logger.info(f'loaded {len(test_dataset)} test data instances')
        
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