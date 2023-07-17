"""REINFORCE algorithm
   Policy network
"""
import torch
import torch.nn as nn
import numpy as np
from ModelFile.loader import load_multihot_problem_to_skill


class Policy_network(nn.Module):
    def __init__(self, args):
        super(Policy_network, self).__init__()
        self.device = args.device
        self.action_num = args.action_num
        self.num_layers = 1
        self.embedding_q = args.embedding_q
        self.embedding_q.weight.requires_grad = False
        self.embed_dim = args.embed_dim
        self.hidden_size = 2 * self.embed_dim

        self.dnn = nn.Linear(self.embed_dim, self.embed_dim)

        self.mlp = nn.Linear(self.action_num, self.embed_dim)

        self.dnn1 = nn.Linear(5 * self.embed_dim, 3 * self.embed_dim)
        self.dnn2 = nn.Linear(3 * self.embed_dim, 3 * self.embed_dim)
        self.dnn3 = nn.Linear(3 * self.embed_dim, 2 * self.embed_dim)
        self.dnn4 = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.fusion_qs = Fusion_Module(self.embed_dim, args.device)
        self.fusion_qsm = Fusion_Module(2 * self.embed_dim, args.device)
        self.lstm = nn.LSTM(4 * self.embed_dim, self.hidden_size, num_layers=self.num_layers, dropout=0, batch_first=True)

        self.linear = nn.Linear(3 * self.embed_dim, 2 * self.embed_dim)
        self.hidden1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.hidden2 = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.val_predict = nn.Linear(self.embed_dim, 1)
        self.predict = nn.Linear(self.embed_dim, 2)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, question_list, answer_list, predict_question, step, mask_list):

        que_list = self.embedding_q(question_list)

        # Fuse question-answer interaction pairs
        q_a_list = self.fusion_qs(que_list, answer_list)  # shape -> [bsz,seq_len,2*emb_dim]
        q_a_mask_list = self.fusion_qsm(q_a_list, mask_list)  # shape -> [bsz,seq_len,4*emb_dim]
        q_a_mask_list = q_a_mask_list[:, :step, :]

        h0 = torch.zeros(self.num_layers, q_a_list.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, q_a_list.size(0), self.hidden_size).to(self.device)
        if step != 0:
            # Get the final knowledge state
            h_t, _ = self.lstm(q_a_mask_list, (h0, c0))  # shape -> [bsz,seq_len,2*emb_dim]
        else:
            h_t = torch.zeros([q_a_list.size(0), q_a_list.size(1), q_a_list.size(2)], dtype=torch.float).to(self.device)

        # 1. Information from the previous state
        h_t = h_t[:, -1, :]  # shape -> [bsz,2*emb_dim]
        # 2. Information from the interaction pair to be removed
        selected_q_a = q_a_list[:, step, :]  # shape -> [bsz,2*emb_dim]
        # 3. Information for predicting the skill
        ques = self.embedding_q(predict_question).view(predict_question.size(0), -1)  # shape -> [bsz,emb_dim]

        state = torch.cat((h_t, selected_q_a), dim=-1)  # shape -> [bsz,4*emb_dim]
        state2 = torch.cat((state, ques), dim=-1)  # shape -> [bsz,5*emb_dim]

        state2 = self.dnn1(state2)  # shape -> [bsz,3*emb_dim]
        state2 = torch.relu(state2)  # shape -> [bsz,3*emb_dim]
        state2 = self.dropout(state2)  # shape -> [bsz,3*emb_dim]

        state2 = self.dnn2(state2)  # shape -> [bsz,3*emb_dim]
        state2 = torch.relu(state2)  # shape -> [bsz,3*emb_dim]
        state2 = self.dropout(state2)  # shape -> [bsz,3*emb_dim]

        state2 = self.dnn3(state2)  # shape -> [bsz,2*emb_dim]
        state2 = torch.relu(state2)  # shape -> [bsz,2*emb_dim]
        state2 = self.dropout(state2)  # shape -> [bsz,2*emb_dim]

        state2 = self.dnn4(state2)  # shape -> [bsz,emb_dim]
        state2 = torch.relu(state2)  # shape -> [bsz,emb_dim]
        state2 = self.dropout(state2)  # shape -> [bsz,emb_dim]

        predict = self.predict(state2)  # shape -> [bsz,2]
        predict = torch.softmax(predict, dim=-1)
        return predict


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
