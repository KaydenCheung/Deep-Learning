import numpy as np
import itertools
import torch
from torch.utils.data.dataset import Dataset


def read_data(path, MAX_STEPS):
    train_qus = np.array([])
    train_ans = np.array([])
    with open(path, 'r') as f:
        for length, ques, ans in itertools.zip_longest(*[f]*3):
            length = int(length.strip())
            ques = np.array(ques.strip().strip(',').split(',')).astype(int)
            ans = np.array(ans.strip().strip(',').split(',')).astype(int)

            mod = 0 if length % MAX_STEPS == 0 else (MAX_STEPS - length % MAX_STEPS)
            zeros = np.zeros(mod) - 1

            ques = np.append(ques, zeros)
            ans = np.append(ans, zeros)

            train_qus = np.append(train_qus, ques)
            train_ans = np.append(train_ans, ans)

    return train_qus.reshape(-1, MAX_STEPS), train_ans.reshape(-1, MAX_STEPS)


class DKTDataSet(Dataset):
    def __init__(self, ques, ans, MAX_STEPS, NUM_OF_QUES):
        super(DKTDataSet, self).__init__()
        self.ques = ques
        self.ans = ans
        self.MAX_STEPS = MAX_STEPS
        self.NUM_OF_QUES = NUM_OF_QUES

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        questions = self.ques[idx]
        answers = self.ans[idx]
        return torch.FloatTensor(self.to_one_hot(questions, answers).tolist())

    def to_one_hot(self, questions, answers):
        res = np.zeros([self.MAX_STEPS, 2 * self.NUM_OF_QUES])
        for i in range(self.MAX_STEPS):
            if answers[i] > 0:
                res[i][int(questions[i])] = 1
            elif answers[i] == 0:
                res[i][int(questions[i])+self.NUM_OF_QUES] = 1
        return res