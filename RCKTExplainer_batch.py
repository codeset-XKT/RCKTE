import math
import torch
import torch.nn as nn
import numpy as np
from ModelFile.model import DKT
from ModelFile.model import SAKT
from policy_network_batch import Policy_network
from torch.distributions import Categorical
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_loader import MyDataset

class RCKTExplainer(nn.Module):
    def __init__(self, args):
        super(RCKTExplainer, self).__init__()
        self.args = args
        self.epoch = 0
        # Configuration for explaining KT model
        if args.model == 'DKT':
            self.KTModel = DKT(self.args).to(self.args.device)
        else:
            self.KTModel = SAKT(self.args).to(self.args.device)
        self.KTModel.load_state_dict(torch.load(self.args.path))
        self.KTModel.eval()
        args.embedding_q = self.KTModel.embedding_q

        if args.policy_type == 'RCKTE':
            # Policy network for maximizing positive effects
            self.policy_p = Policy_network(args=args)
            # Policy network for maximizing negative effects
            self.policy_n = Policy_network(args=args)

        self.optimizer_p = torch.optim.Adam(self.policy_p.parameters(), lr=args.RL_lr,
                                            weight_decay=args.l2_weight)
        self.optimizer_n = torch.optim.Adam(self.policy_n.parameters(), lr=args.RL_lr,
                                            weight_decay=args.l2_weight)

        self.policy_p.to(args.device)
        self.policy_n.to(args.device)

    # KT model prediction
    def predict(self, question_list, answer_list, predict_question, is_base):
        with torch.no_grad():
            if is_base:
                pad_predict = self.KTModel(question_list, answer_list,
                                           torch.cat((question_list[:, 1:], predict_question), dim=-1))
            else:
                pad_predict = self.KTModel(question_list, answer_list, predict_question)
        return pad_predict

    # Run one episode
    def run_episode(self, base_val, question_list, answer_list, predict_question, direction):
        action_list = []
        state_list = []
        prob_list = []
        predict_list = []
        baseline_list = []
        action_pro_list = []

        mask_list = torch.randint(1, 2, (question_list.size(0) , self.args.action_num)).to(self.args.device)  # [bsz, action_num]

        for step in range(self.args.action_num):
            # Get the next action
            if direction == 'P':
                action_pro = self.policy_p(question_list, answer_list, predict_question, step, mask_list)  # shape -> [bsz,2]
            else:
                action_pro = self.policy_n(question_list, answer_list, predict_question, step, mask_list)  # shape -> [bsz,2]

            action_pro_list.append(action_pro)

            action_index, log_prob = self.select_action(action_pro)  # [bsz]
            action_list.append(action_index)
            state_list.append((question_list, mask_list))
            prob_list.append(log_prob)

            # Update the mask sequence to get the predicted value
            mask_list[:, step] = action_index
            nan_mask_list = mask_list.clone().to(torch.float64)

            # Calculate the filled positions
            filled_index = torch.sum(nan_mask_list, dim=-1).to(torch.long)  # [bsz]
            filled_index = filled_index - 1

            nan_mask_list[nan_mask_list == 0] = np.nan
            item = nan_mask_list * question_list  # [bsz, action_num]
            item_cleaned = []
            answer = nan_mask_list * answer_list
            answer_cleaned = []
            prediction_question_list_cleaned = []

            for row_item, row_answer, row_predict in zip(item, answer, predict_question):
                row_cleaned_item = [elem for elem in row_item.tolist() if not math.isnan(elem)]
                row_cleaned_answer = [elem for elem in row_answer.tolist() if not math.isnan(elem)]
                prediction_question_list = row_cleaned_item.copy()
                prediction_question_list.append(row_predict.item())
                prediction_question_list = prediction_question_list[1:]
                padding_length = self.args.action_num - len(row_cleaned_item)
                row_cleaned_item += [0] * padding_length
                row_cleaned_answer += [0] * padding_length
                prediction_question_list += [0] * padding_length
                item_cleaned.append(row_cleaned_item)
                answer_cleaned.append(row_cleaned_answer)
                prediction_question_list_cleaned.append(prediction_question_list)
            filled_item = torch.tensor(item_cleaned).to(torch.long).cuda()
            filled_answer = torch.tensor(answer_cleaned).to(torch.long).cuda()
            filled_prediction_list = torch.tensor(prediction_question_list_cleaned).to(torch.long).cuda()

            predict_val = self.predict(filled_item, filled_answer, filled_prediction_list, False)
            mask_val = predict_val[torch.arange(0, predict_val.size(0)), filled_index].view(-1)  # [bsz]

            # When mask_list is all 0, mask_val should be equal to base_val
            none_index = torch.nonzero(filled_index == -1).squeeze()
            mask_val[none_index] = base_val[none_index]

            predict_list.append(mask_val)

        predict_list = torch.stack(predict_list)

        if direction == 'N':
            ice_list = - (predict_list - base_val)
        else:
            ice_list = predict_list - base_val  # [action_num, bsz]

        # Calculate the reward/return for each batch
        diff_tensor = torch.diff(ice_list, dim=0)
        reward_tensor = torch.cat((ice_list[:1], diff_tensor), dim=0)
        reward_list = torch.transpose(reward_tensor, 0, 1)  # [bsz, action_num]
        reward_list = reward_list.tolist()

        if self.args.isReturn:
            # Discounted return
            alpha = 0.995
            ut_list = []
            for batch_reward_list in reward_list:
                batch_ut_list = []
                for i in range(len(batch_reward_list)):
                    r = 0
                    for idx, j in enumerate(range(i, len(batch_reward_list))):
                        r += math.pow(alpha, idx) * batch_reward_list[j]
                    batch_ut_list.append(r)
                ut_list.append(batch_ut_list)  # [bsz,action_num]
        else:
            ut_list = reward_list

        # Convert to batch shape
        prob_list = torch.stack(prob_list)
        action_list = torch.stack(action_list)
        action_pro_list = torch.stack(action_pro_list)

        prob_list = torch.transpose(prob_list, 0, 1)
        action_list = torch.transpose(action_list, 0, 1)
        action_pro_list = torch.transpose(action_pro_list, 0, 1)

        repisode = sum(list(map(lambda x: sum(x), ut_list)))

        return prob_list, predict_list, action_list, baseline_list, action_pro_list, ut_list, repisode

    # Action selection
    def select_action(self, action_pro):
        dist = Categorical(action_pro)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        # if np.random.rand() < 0.5 * np.power(0.995, self.epoch):
        #     action_index = torch.randint(2, (action_pro.size(0),)).cuda()
        return action_index, log_prob  # [bsz,1]

    # Update parameters
    def update_parameters(self, ut_list, prob_list, direction):
        loss = 0
        for batch_ut_list, batch_pro_list in zip(ut_list, prob_list):
            for batch_ut, batch_log_prob in zip(batch_ut_list, batch_pro_list):
                loss += -batch_ut * batch_log_prob
        # Backpropagate and update the value network gradients
        if direction == 'P':
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_p.step()
        else:
            self.optimizer_n.zero_grad()
            loss.backward()
            self.optimizer_n.step()

    # Train the model
    def run(self):
        r_epoch_list_p = []
        r_epoch_list_n = []
        r_ice_list = []
        logodds_epoch_list = []
        dataset = MyDataset('{0}\\{1}\\train_test\\train_question.txt'.format(self.args.data_path, self.args.data_set), self.args.action_num)
        if len(dataset) >= 3000:
            indices = range(3000)  # Take the first 3000 data points
            dataset = torch.utils.data.Subset(dataset, indices)
        dataset_len = len(dataset)
        print("Training dataset size:", dataset_len)
        dataloader = DataLoader(dataset, batch_size=self.args.RL_BSZ, shuffle=True)
        max_test_ice = 0
        best_epoch = 0

        for epoch in range(self.args.RL_epoch):
            self.epoch += 1
            # Train policy gradients/value networks
            self.policy_p.train()
            self.policy_n.train()
            r_epoch_p = 0
            r_epoch_n = 0
            ice_epoch = 0
            idx = 0
            for batch in tqdm(dataloader):
                idx += 1
                question_list, answer_list, predict_question, predict_answer = batch
                base_val = self.predict(question_list, answer_list, predict_question, True)[:, -1]  # [bsz]

                # Execute one episode - positive effects
                prob_list, predict_list, action_list, baseline_list, action_pro_list, ut_list, repisode = self.run_episode(
                    base_val, question_list, answer_list, predict_question, 'P')
                # Update parameters and get the episode reward
                r_epoch_p += repisode
                self.update_parameters(ut_list, prob_list, 'P')
                predict_p = predict_list[-1, :]  # [bsz] Final predicted result of the interpretable subsequence
                result_p = torch.abs(predict_p - base_val)  # [bsz]

                # Execute one episode - negative effects
                prob_list, predict_list, action_list, baseline_list, action_pro_list, ut_list, repisode = self.run_episode(
                    base_val, question_list, answer_list, predict_question, 'N')
                # Update parameters and get the episode reward
                r_epoch_n += repisode
                self.update_parameters(ut_list, prob_list, 'N')
                predict_n = predict_list[-1, :]  # [bsz] Final predicted result of the interpretable subsequence
                result_n = torch.abs(predict_n - base_val)  # [bsz]

                # < - Calculate ICE value - >
                for max_p, max_n in zip(result_p, result_n):
                    if max_p > max_n:
                        ice_epoch += max_p
                    else:
                        ice_epoch += max_n
                # < - Calculate ICE value - >

                if idx == 1:
                    print("action_pro_list", action_pro_list[0].tolist())
                    print("action_list", action_list[0])

            r_epoch_list_p.append(r_epoch_p)
            r_epoch_list_n.append(r_epoch_n)
            r_ice_list.append(ice_epoch.item())

            print("Epoch", epoch, ", positive return:", r_epoch_p, ", negative return:", r_epoch_n)
            print("Epoch", epoch, ", total ICE:", ice_epoch.item())

            # Get performance on the test set
            r_p_test, r_n_test, r_ice_test, logodds_list = self.test()
            logodds_epoch_list.append(logodds_list)
            if r_ice_test > max_test_ice:
                best_epoch = epoch
                max_test_ice = r_ice_test


        print("Maximum ICE value on the test set:", max_test_ice)
        print("Corresponding log odds value:", sum(logodds_epoch_list[best_epoch]))

    # Performance on the test set based on ICE
    def test(self):

        # Train policy gradients/value networks
        self.policy_p.eval()
        self.policy_n.eval()
        r_epoch_p = 0
        r_epoch_n = 0
        ice_epoch = 0
        mask_length = 0
        # Distribution of log odds
        logodds_list = []

        dataset = MyDataset('{0}\\{1}\\train_test\\test_question.txt'.format(self.args.data_path, self.args.data_set), self.args.action_num)
        if len(dataset) >= 300:
            indices = range(300)  # Take the first 300 data points
            dataset = torch.utils.data.Subset(dataset, indices)
        dataset_len = len(dataset)
        print("Test dataset size:", dataset_len)
        dataloader = DataLoader(dataset, batch_size=self.args.RL_BSZ, shuffle=True)
        idx = 0
        for batch in dataloader:
            idx += 1
            question_list, answer_list, predict_question, predict_answer = batch
            base_val = self.predict(question_list, answer_list, predict_question, True)[:, -1]  # [bsz]

            # Execute one episode - positive effects
            prob_list_p, predict_list_p, action_list_p, baseline_list_p, action_pro_list_p, ut_list_p, repisode_p = self.run_episode(
                base_val, question_list, answer_list, predict_question, 'P')
            r_epoch_p += repisode_p
            # predict_list_p -> shape [action_num,bsz]
            predict_p = predict_list_p[-1, :]  # [bsz] Final predicted result of the interpretable subsequence
            result_p = torch.abs(predict_p - base_val)  # [bsz]

            # Execute one episode - negative effects
            prob_list, predict_list, action_list_n, baseline_list, action_pro_list, ut_list, repisode = self.run_episode(
                base_val, question_list, answer_list, predict_question, 'N')
            r_epoch_n += repisode
            predict_n = predict_list[-1, :]  # [bsz] Final predicted result of the interpretable subsequence
            result_n = torch.abs(predict_n - base_val)  # [bsz]

            # < - Calculate ICE value - >
            for max_p, max_n, action_p, action_n in zip(result_p, result_n, action_list_p, action_list_n):
                if max_p > max_n:
                    ice_epoch += max_p
                    mask_length += self.args.action_num - torch.sum(action_p)
                else:
                    ice_epoch += max_n
                    mask_length += self.args.action_num - torch.sum(action_n)
            # < - Calculate ICE value - >

            # < - Calculate Log Odds ->
            log_p = torch.log(predict_p/(1-predict_p))
            log_n = torch.log(predict_n/(1-predict_n))
            log_base = torch.log(base_val/(1-base_val))
            logodds_p = log_base - log_p
            logodds_n = log_base - log_n
            for odds_p, odds_n in zip(logodds_p, logodds_n):
                if torch.abs(odds_p) > torch.abs(odds_n):
                    logodds_list.append(odds_p.item())
                else:
                    logodds_list.append(odds_n.item())
            # < - Calculate Log Odds ->

            if idx == 1:
                print("Test set, action_pro_list", action_pro_list[0][0].tolist())
                print("Test set, action_list", action_list_n[0].tolist())

        print("Test set, positive return:", r_epoch_p, ", negative return:", r_epoch_n)
        print("Test set, odds:", sum(list(map(lambda x: abs(x), logodds_list))))
        print("Test set, total ICE:", ice_epoch.item())
        print("Test set, total length:", mask_length.item())
        print("Test set, unit ICE explanation length:", (mask_length/ice_epoch).item(), '\n')

        return r_epoch_p, r_epoch_n, ice_epoch.item(), logodds_list


# Data parameter configuration
def parse_args():
    # Select dataset and model
    embed_type = 'QE'
    data_set = 'EdNet'
    model = 'SAKT'
    action_num = 20

    # Number of questions in the dataset
    question_num_dict = {
        'ASSIST09': 15700,
        'EdNet': 11848,
        'ASSIST12': 47000,
        'JunYi': 669
    }

    # Corresponding pkl file for the dataset model, QE for question_embedding
    pkl_dict = {
        "QE": {
            "DKT_ASSIST09": 'DKT_ASSIST09_QE_0.737.pkl',
            "DKT_EdNet": 'DKT_EdNet_QE_0.756.pkl',
            "SAKT_ASSIST09": 'SAKT_ASSIST09_QE_0.751.pkl',
            "SAKT_EdNet": 'SAKT_EdNet_QE_0.757.pkl',
            "DKT_ASSIST12": 'DKT_ASSIST12_QE_0.752.pkl',
            "SAKT_ASSIST12": 'SAKT_ASSIST12_QE_0.753.pkl',
            "DKT_JunYi": 'DKT_JunYi_QE_0.782.pkl',
            "SAKT_JunYi": 'SAKT_JunYi_QE_0.784.pkl'
        }
    }

    parser = argparse.ArgumentParser()
    # for DKT
    parser.add_argument('--device', type=str,
                        default="cuda:0")
    parser.add_argument("--input", type=str,
                        default='question')
    parser.add_argument("--data_path", type=str,
                        default="./data")
    parser.add_argument("--question_num", type=str,
                        default=question_num_dict[data_set])
    parser.add_argument("--data_set", type=str,
                        default=data_set)
    parser.add_argument("--model", type=str,
                        default=model)
    parser.add_argument("--embed_dim", type=int,
                        default=128)

    # for SAKT
    parser.add_argument('--num_attn_layer', type=int, default=2)  # 2 or 4
    parser.add_argument('--num_heads', type=int, default=2)  # 2 or 4
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=256)
    parser.add_argument('--drop_prob', type=float, default=0.05)

    # for RCKTExplainer
    parser.add_argument("--embed_type", type=str,
                        default=embed_type)
    parser.add_argument("--path", type=str,
                        default='./param/{0}'.format(pkl_dict[embed_type][model+'_'+data_set]))
    # Length of the decision selection
    parser.add_argument("--action_num", type=int,
                        default=action_num)
    # Use discounted return or reward return (greedy strategy)
    parser.add_argument("--isReturn", type=bool,
                        default=False)
    # Type of policy network RCKTE or simple MLP
    parser.add_argument("--policy_type", type=str,
                        default='RCKTE')
    parser.add_argument("--RL_BSZ", type=int,
                        default=32)
    parser.add_argument("--RL_epoch", type=int,
                        default=1000)
    parser.add_argument("--RL_lr", type=int,
                        default=5e-6)
    parser.add_argument("--l2_weight", type=int,
                        default=1e-5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    explainer = RCKTExplainer(args)
    explainer.run()
