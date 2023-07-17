import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, file_path, sequence_length):
        self.data = self.load_data(file_path, sequence_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index]
        question_list = torch.tensor(sequence[0]).cuda()
        answer_list = torch.tensor(sequence[1]).cuda()
        predict_question = torch.tensor(sequence[2]).cuda()
        predict_answer = torch.tensor(sequence[3]).cuda()
        return question_list, answer_list, predict_question, predict_answer

    def load_data(self, file_path, sequence_length):
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 3):
                sequence_length_actual = int(lines[i].strip())
                if sequence_length_actual >= sequence_length+1:
                    question_list = list(map(int, lines[i+1].strip().split(',')))
                    answer_list = list(map(int, lines[i+2].strip().split(',')))
                    predict_question = [question_list[sequence_length]]
                    predict_answer = [answer_list[sequence_length]]
                    data.append((question_list[:sequence_length], answer_list[:sequence_length], predict_question, predict_answer))
        return data