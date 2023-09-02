import os
import torch
import time as T
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.model import ISDFormer
from dataloader.dataloader import Dataset_Custom
import utils.tools as tools


class ISDFormer_exp:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.batch_size = args.batch_size
        self.model = ISDFormer(self.args).to(self.device)
        self.crit = nn.MSELoss()
        self.optim = self.select_optim()
        self.checkpoint_path = os.path.join('./checkpoint/', args.checkpoint)
        self.logfile_path = 'logfile/'
        self.scaler = tools.StandardScaler()

    def select_optim(self):
        return torch.optim.Adam(self.model.parameters(), self.args.learning_rate)

    def train(self):
        self.model.train()
        ds_cus = Dataset_Custom(self.args, flag='train')
        power_loader = DataLoader(dataset=ds_cus, batch_size=self.batch_size, shuffle=True, drop_last=True)
        logfile_name = os.path.join(self.logfile_path, str(T.strftime('%Y%m%d %H.%M.%S', T.localtime())) + '.txt')

        total_loss = []
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        if self.args.log:
            logfile = open(logfile_name, 'a')
            logfile.write('-----------------------------------------------------config '
                          'info-----------------------------------------------------\n')
            for i in self.args.__dict__:
                logfile.write(i)
                logfile.write(':   ')
                logfile.write(str(self.args.__dict__[i]))
                logfile.write('\n')
            logfile.write('-----------------------------------------------------training '
                          'info-----------------------------------------------------\n')
            logfile.close()
        for i in range(self.args.train_epochs):
            print("epoch ", i + 1, ": ")
            start = T.time()
            epochloss = []
            for j, (seq, seq_mark, pred_mark, ground_truth) in enumerate(power_loader):
                self.optim.zero_grad()
                pred_result, ground_truth = self.process_one_batch(seq, seq_mark, pred_mark, ground_truth)
                loss = self.crit(pred_result, ground_truth)
                loss.backward()
                self.optim.step()
                if j % 50 == 0:
                    print("itr ", j, "/", int(len(power_loader)), " Loss: ", loss.item())
                    if self.args.log:
                        logfile = open(logfile_name, 'a')
                        logfile.writelines(
                            ["epoch ", str(i + 1), ": ""itr ", str(j), "/", str(int(len(power_loader))), " Loss: ",
                             str(loss.item()), '\n'])
                        logfile.close()
                epochloss.append(loss.item())
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print("----------test------------")
            test_mse, test_mae = self.test()
            end = T.time()
            epoch_avg_loss = np.mean(epochloss)
            total_loss.append(epoch_avg_loss)
            print("Loss: ", epoch_avg_loss)
            print("Cost Time:", round((end - start) / 60, 2), " minutes")
            print("Left Time:", round((end - start) / 60 * (self.args.train_epochs - i), 2), "minutes")
            if self.args.log:
                logfile = open(logfile_name, 'a')
                logfile.writelines(
                    ["Loss: ", str(epoch_avg_loss), '\n', "Cost Time:", str(round((end - start) / 60, 2)), " minutes",
                     '\n'])
                logfile.writelines(
                    ["Test MSE: ", str(test_mse), '\n', "Test MAE: ", str(test_mae), '\n'])
                logfile.close()
        plt.plot(total_loss)
        plt.savefig(os.path.join(self.logfile_path, "loss.jpg"))
        plt.show()

    def pred(self, index):
        args = self.args
        args.batch_size = 1
        self.model.eval()
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        else:
            raise 'The model has not be train!'
        ds_cus = Dataset_Custom(args, flag='val')
        seq, seq_mark, pred_mark, ground_truth = ds_cus[index]
        seq = seq.unsqueeze(0)
        seq_mark = seq_mark.unsqueeze(0)
        pred_mark = pred_mark.unsqueeze(0)
        ground_truth = ground_truth.unsqueeze(0)

        pred_result, ground_truth = self.process_one_batch(seq, seq_mark, pred_mark, ground_truth)
        pred_result = torch.cat((seq[:, :, -1:], pred_result.cpu()), dim=1)
        ground_truth = torch.cat((seq[:, :, -1:], ground_truth.cpu()), dim=1)

        plt.plot(pred_result[0, :, -1].cpu().detach().numpy(), label="Pred", linewidth=2)
        plt.plot(ground_truth[0, :, -1].cpu().detach().numpy(), label="Gt", linewidth=2)
        plt.legend()
        plt.show()

    def test(self):
        args = self.args
        self.model.eval()
        ds_cus = Dataset_Custom(args, flag='val')
        power_loader = DataLoader(dataset=ds_cus, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        else:
            raise 'The model has not be train!'
        MSE = []
        MAE = []
        with torch.no_grad():
            for j, (seq, seq_mark, pred_mark, ground_truth) in enumerate(power_loader):

                pred_result, ground_truth = self.process_one_batch(seq, seq_mark, pred_mark, ground_truth)

                MSE.append(tools.MSE(ground_truth.cpu().detach().numpy(),
                                     pred_result.cpu().detach().numpy()))
                MAE.append(tools.MAE(ground_truth.cpu().detach().numpy(),
                                     pred_result.cpu().detach().numpy()))
            print("MSE: ", np.mean(MSE))
            print("MAE: ", np.mean(MAE))

            return np.mean(MSE), np.mean(MAE)

    def process_one_batch(self, seq, seq_mark, pred_mark, ground_truth):
        seq = seq.to(self.device)
        seq_mark = seq_mark.to(self.device)
        pred_val = torch.zeros(self.args.batch_size, self.args.pred_len, 1).to(self.args.device)
        pred_mark = pred_mark.to(self.device)
        ground_truth = ground_truth.to(self.device)
        outputs = self.model(seq, seq_mark, pred_val, pred_mark)

        return outputs, ground_truth
