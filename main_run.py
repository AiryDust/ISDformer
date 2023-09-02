import argparse
from exp.ISDFormer_EXP import ISDFormer_exp

parser = argparse.ArgumentParser(description='[ISDFormer Sequences Forecasting]')

# ---------------------------------data config-------------------------------------#
parser.add_argument('--root_path', type=str, default=r'./data', help='path of the dataset')
parser.add_argument('--data_name', type=str, default='solar.csv',
                    help='specified format. See ./dataloader/dataloader.py')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='data frequency of sample')
parser.add_argument('--scale', type=bool, default=True, help='Whether to standardize the dataset')

# ---------------------------------ISDFormer config----------------------------------#
parser.add_argument('--seq_len', type=int, default=96*2, help='input sequence length of encoder')
parser.add_argument('--pred_len', type=int, default=96*1, help='prediction sequence length')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[S, MS]; S:univariate '
                                                               'predict univariate, MS:multivariate predict univariate')
parser.add_argument('--feature_dimension', type=int, default=8, help='the num of parm dimension without date.')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of conv function')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument '
                                                           'means not using distilling', default=False)

parser.add_argument('--dropout', type=float, default=0.35, help='dropout')
parser.add_argument('--embed_type', type=str, default='learned',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--window', type=str, default='rect',
                    help='shape of the frequency window. options:[hamming, hanning, rect]')
parser.add_argument('--window_length', type=int, default=96, help='length of the window. usually be one period')
parser.add_argument('--stride', type=int, default=1, help='length of the window stride')

# ----------------------------------training config-----------------------------#
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--checkpoint', type=str, default='Solar.pt', help='checkpoint name')
parser.add_argument('--log', type=bool, default=True, help='whether need logfile')
parser.add_argument('--device', type=str, default='cuda', help='device to run the model, option:[cuda, cpu]')

args = parser.parse_args()

# 生活处处皆苦难
# 人生处处皆坎坷
if __name__ == '__main__':
    exp = ISDFormer_exp(args)
    print(">>>>>>>>>>>>>>>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>")
    exp.train()
    print(">>>>>>>>>>>>>>>>>>>>>start test>>>>>>>>>>>>>>>>>>>>>>>>>>")
    exp.test()
    exp.pred(484)

