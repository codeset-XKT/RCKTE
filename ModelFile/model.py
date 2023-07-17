import torch
import torch.nn as nn
import os

from ModelFile.loader import load_multihot_problem_to_skill
from ModelFile.SAKT import SAN_SAKT

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Always report errors for shape inconsistency below

class DKT(nn.Module):
    """
    Model: DKT
    Author: yuanxin
    """

    def __init__(self, args):
        super(DKT, self).__init__()
        # self.dict, self.embedding = load_multihot_problem_to_skill(args)  # Multi-hot encoding of skills for questions
        self.embedding_q = nn.Embedding(args.question_num, args.embed_dim)
        # self.embed_dim = len(self.embedding[0])
        self.embed_dim = args.embed_dim
        self.fusion = Fusion_Module(self.embed_dim, args.device)

        self.device = args.device
        self.hidden_size = self.embed_dim
        self.num_layers = 1

        self.lstm = nn.LSTM(2 * self.embed_dim, self.hidden_size, num_layers=self.num_layers, dropout=0,
                            batch_first=True)
        self.hidden = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.predict = nn.Linear(self.embed_dim, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, q, a, next_q):

        q = self.embedding_q(q)
        next_q = self.embedding_q(next_q)   # [batch_size, length, embed_dim]

        # Fuse skill encoding with answer
        x = self.fusion(q, a)
        # x.requires_grad_()

        # LSTM layer for updating knowledge state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        stu_state, _ = self.lstm(x, (h0, c0))  # [batch_size, length, 2*embed_dim]

        # Prediction layer
        y = torch.cat((stu_state, next_q), dim=-1)
        y = self.hidden(y)
        y = torch.relu(y)
        y = self.predict(y)
        y = self.dropout(y)
        y = torch.sigmoid(y).squeeze(-1)
        return y


class SAKT(nn.Module):
    """
    Skill embedding
    Model: SAKT
    Author: yuanxin
    """

    def __init__(self, args):
        super(SAKT, self).__init__()
        # self.dict, self.embedding_s = load_multihot_problem_to_skill(args)  # Multi-hot encoding of skills for questions
        self.embed_dim = args.embed_dim
        self.embedding_q = nn.Embedding(args.question_num, self.embed_dim)
        # self.embed_dim = len(self.embedding[0])
        self.ks_module = SAN_SAKT(args, self.embed_dim)
        self.fusion = Fusion_Module(self.embed_dim, args.device)
        self.concat = nn.Linear(self.embed_dim, self.embed_dim)
        self.hidden = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.predict = nn.Linear(self.embed_dim, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.device = args.device

    def forward(self, q, a, next_q):
        q = self.embedding_q(q)
        next_q = self.embedding_q(next_q)

        # Fuse question encoding with answer
        q_a = self.fusion(q, a)
        ks_emb = self.ks_module(q, q_a, next_q)

        # Prediction layer
        y = torch.cat((ks_emb, next_q), dim=-1)
        y = self.hidden(y)
        y = torch.relu(y)
        y = self.predict(y)
        y = self.dropout(y)
        y = torch.sigmoid(y).squeeze(-1)
        return y


class Fusion_Module(nn.Module):
    def __init__(self, emb_dim, device):
        super(Fusion_Module, self).__init__()
        self.transform_matrix = torch.zeros(2, emb_dim * 2).to(device)
        self.transform_matrix[0][emb_dim:] = 1.0
        self.transform_matrix[1][:emb_dim] = 1.0

    def forward(self, ques_emb, pad_answer):
        ques_emb = torch.cat((ques_emb, ques_emb), -1)
        answer_emb = nn.functional.embedding(pad_answer, self.transform_matrix)
        input_emb = ques_emb * answer_emb

        return input_emb
