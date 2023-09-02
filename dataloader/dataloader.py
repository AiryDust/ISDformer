import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.tools import StandardScaler
from utils.timefeatures import time_features



class Dataset_Custom(Dataset):
    def __init__(self, args, flag='train'):

        self.features = args.features
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.target = args.target
        self.scale = args.scale
        self.timeenc = 1
        self.freq = args.freq
        self.root_path = args.root_path
        self.data_path = args.data_name
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        singleData_len = (self.seq_len + self.pred_len)

        border1s = [0, num_train, num_train + num_test]
        border2s = [num_train - singleData_len, num_train + num_test - singleData_len,
                    len(df_raw) - singleData_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]


        if self.scale:
            data = df_data[border1:border2]
            self.scaler.fit(data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2 + self.pred_len]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        if self.features == 'S':
            self.data = data[border1:border2 + self.pred_len, -1:]
        else:
            self.data = data[border1:border2 + self.pred_len]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        seq_begin = index
        seq_end = seq_begin + self.seq_len
        groundTruth_begin = seq_end
        groundTruth_end = groundTruth_begin + self.pred_len

        seq = torch.tensor(self.data[seq_begin:seq_end]).to(torch.float32)
        seq_mark = torch.tensor(self.data_stamp[seq_begin:seq_end]).to(torch.float32)
        pred_mark = torch.tensor(self.data_stamp[groundTruth_begin:groundTruth_end]).to(torch.float32)
        groundTruth = torch.tensor(self.data[groundTruth_begin:groundTruth_end, -1:]).to(torch.float32)

        return seq, seq_mark, pred_mark, groundTruth

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
