import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn import metrics
import argparse

from ModelFile.train_utils import Logger
from ModelFile.model import DKT
from ModelFile.model import SAKT
from ModelFile.loader import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def demo_train(args):
    print("setting: \n\n", args, "\n")
    logger = Logger(args)
    # load data
    loader = load_data(args)

    model = DKT(args)

    model.to(device)

    print(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.l2_weight)
    criterion = nn.BCELoss()
    for epoch in range(args.epochs):

        logger.epoch_increase()
        epoch_loss = 0
        for i, (seq_lens, pad_data, pad_answer, pad_index, pad_label) in enumerate(loader['train']):
            pad_predict = model(pad_data, pad_answer, pad_index)  # [bsz,seq_len]
            pack_predict = pack_padded_sequence(pad_predict, seq_lens.to("cpu"), enforce_sorted=True,
                                                batch_first=True)  # [bsz*seq_len]
            pack_label = pack_padded_sequence(pad_label, seq_lens.to("cpu"), enforce_sorted=True,
                                              batch_first=True)  # [bsz*seq_len]

            loss = criterion(pack_predict.data, pack_label.data)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics_dict = evaluate(model, loader['train'])
        test_metrics_dict = evaluate(model, loader['test'])

        logger.one_epoch(epoch, train_metrics_dict, test_metrics_dict, model)

        if logger.is_stop():
            break
        print(epoch, "epoch_loss", epoch_loss)

    logger.one_run(args)

def evaluate(model, data):
    model.eval()
    true_list, pred_list = [], []
    for seq_lens, pad_data, pad_answer, pad_index, pad_label in data:
        pad_predict = model(pad_data, pad_answer, pad_index)  # 运行模型
        pack_predict = pack_padded_sequence(pad_predict, seq_lens.to("cpu"), enforce_sorted=True,
                                            batch_first=True)  # [bsz*seq_len]
        pack_label = pack_padded_sequence(pad_label, seq_lens.to("cpu"), enforce_sorted=True,
                                          batch_first=True)  # [bsz*seq_len]

        y_true = pack_label.data.cpu().contiguous().view(-1).detach()
        y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()

        true_list.append(y_true)
        pred_list.append(y_pred)

    all_pred = torch.cat(pred_list, 0)
    all_target = torch.cat(true_list, 0)
    auc = metrics.roc_auc_score(all_target, all_pred)

    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    acc = metrics.accuracy_score(all_target, all_pred)

    model.train()
    return {'auc': auc, 'acc': acc}

# 数据参数配置
def parse_args(data_set):

    question_num_dict = {
        'ASSIST09': 15700,
        'EdNet': 11848,
        'ASSIST12': 47000,
        'JunYi': 669
    }

    parser = argparse.ArgumentParser()
    # 数据参数配置
    parser.add_argument("--batch_size", type=int,
                        default=32)
    parser.add_argument("--min_seq_len", type=int,
                        default=3)
    parser.add_argument("--max_seq_len", type=int,
                        default=200)
    parser.add_argument('--device', type=str,
                        default="cuda")
    parser.add_argument("--input", type=str,
                        default='question')
    parser.add_argument("--question_num", type=str,
                        default=question_num_dict[data_set])
    parser.add_argument("--data_path", type=str,
                        default="..\data")
    parser.add_argument("--data_set", type=str,
                        default=data_set)
    parser.add_argument('--save_dir', type=str,
                        default='./result/{0}'.format(data_set),
                        help='the dir which save results')
    parser.add_argument('--log_file', type=str,
                        default='logs.txt',
                        help='the name of logs file')
    parser.add_argument('--result_file', type=str,
                        default='tunings.txt',
                        help='the name of results file')
    parser.add_argument('--remark', type=str,
                        default='', help='remark the experiment')
    parser.add_argument("--patience", type=int,
                        default=10)
    # for SAKT
    parser.add_argument('--num_attn_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=256)
    parser.add_argument('--drop_prob', type=float, default=0.05)

    # 模型参数配置
    parser.add_argument("--epochs", type=int,
                        default=1000)
    parser.add_argument("--embed_dim", type=int,
                        default=128)
    parser.add_argument("--lr", type=int,
                        default=0.001)
    parser.add_argument("--l2_weight", type=int,
                        default=1e-5)

    return parser.parse_args()

args = parse_args('EdNet')

if __name__ == '__main__':
    demo_train(args)
