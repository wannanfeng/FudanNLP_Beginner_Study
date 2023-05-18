import random
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
def data_split(data, test_rate = 0.3):
    train = []
    test = []
    for dataum in data:
        if random.random() > test_rate:
            train.append(dataum)
        else:
            test.append(dataum)
    return train, test

class ClsDataset(Dataset):
    def __init__(self, sentence1, sentence2, type):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.tpye = type
    def __getitem__(self, item):
        return self.sentence1[item], self.sentence2[item], self.tpye[item]
    def __len__(self):
        return len(self.tpye)

def collate_fn(batch_data):
    """ 自定义一个batch里面的数据的组织方式 """
    sent1, sent2, label = zip(*batch_data)
    sentence1 = [torch.LongTensor(sent) for sent in sent1]
    padded1 = pad_sequence(sentence1, batch_first=True, padding_value=0)
    sentence2 = [torch.LongTensor(sent) for sent in sent2]
    padded2 = pad_sequence(sentence2, batch_first=True, padding_value=0)
    return torch.LongTensor(padded1), torch.LongTensor(padded2), torch.LongTensor(label)

def get_batch(x1,x2,y,batch_size):
    dataset = ClsDataset(x1,x2, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False ,drop_last=True, collate_fn=collate_fn)
    return dataloader

class Random_embedding():
    def __init__(self, data, test_rate = 0.3):
        self.dict_words = dict() #创建自己的字典
        _data = [item.split('\t') for item in data]
        self.data = [[item[5],item[6],item[0]] for item in _data]  #item[5]为一个句子 item[6]为一个 item[0]为句子间类型
        self.data.sort(key=lambda x:len(x[0].split()))

        self.train, self.test = data_split(self.data, test_rate=test_rate)
        self.len_words = 0  #字典中单词总数
        self.type_dict = {'-':0, 'contradiction':1, 'entailment':2, 'neutral':3}
        self.train_y = [self.type_dict[item[2]] for item in self.train]  #对应的句子类型转为数字
        self.test_y = [self.type_dict[item[2]] for item in self.test]

        self.train_s1_matrix = list()
        self.test_s1_matrix = list()
        self.train_s2_matrix = list()
        self.test_s2_matrix = list()
        self.longest = 0
    def get_words(self):
        pattern = '[A-Za-z|\']+' #正则表达式
        for item in self.data:
            for i in range(2):
                s = item[i]
                s = s.upper()
                words = re.findall(pattern, s) #字符串中查找所有匹配指定模式的子串，并返回一个列表
                for word in words:
                    if word not in self.dict_words:
                        self.dict_words[word] = len(self.dict_words) + 1
        self.len_words = len(self.dict_words)
    def get_id(self):
        pattern = '[A-Za-z|\']+'
        for item in self.train:
            for i in range(2):
                s = item[i]
                s = s.upper()
                words = re.findall(pattern, s)
                term = [self.dict_words[word] for word in words]
                self.longest = max(self.longest, len(term))
                if i == 0:
                    self.train_s1_matrix.append(term)
                else:
                    self.train_s2_matrix.append(term)
        for item in self.test:
            for i in range(2):
                s = item[i]
                s = s.upper()
                words = re.findall(pattern, s)
                term = [self.dict_words[word] for word in words]
                self.longest = max(self.longest, len(term))
                if i == 0:
                    self.test_s1_matrix.append(term)
                else:
                    self.test_s2_matrix.append(term)
        self.len_words += 1

class Glove_embedding():
    def __init__(self, data, trained_dict, test_rate=0.3):
        self.dict_words = dict()
        _data = [item.split('\t') for item in data]
        self.data = [[item[5], item[6], item[0]] for item in _data]
        self.data.sort(key=lambda x:len(x[0].split()))
        self.trained_dict = trained_dict #训练好的字典
        self.len_words = 0
        self.train, self.test = data_split(self.data, test_rate=test_rate)
        self.type_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.train_y = [self.type_dict[term[2]] for term in self.train]  # Relation in training set
        self.test_y = [self.type_dict[term[2]] for term in self.test]  # Relation in test set
        self.train_s1_matrix = list()
        self.test_s1_matrix = list()
        self.train_s2_matrix = list()
        self.test_s2_matrix = list()
        self.longest = 0
        self.embedding = list()

    def get_words(self):
        self.embedding.append([0]*50)
        pattern = '[A-Za-z|\']+'
        for term in self.data:
            for i in range(2):
                s = term[i]
                s = s.upper()
                words = re.findall(pattern, s)
                for word in words:  # Process every word
                    if word not in self.dict_words:
                        self.dict_words[word] = len(self.dict_words)
                        if word in self.trained_dict:
                            self.embedding.append(self.trained_dict[word])
                        else:
                            # print(word)
                            # raise Exception("words not found!")
                            self.embedding.append([0] * 50)
        self.len_words = len(self.dict_words)

    def get_id(self):
        pattern = '[A-Za-z|\']+'
        for item in self.train:
            for i in range(2):
                s = item[i]
                s = s.upper()
                words = re.findall(pattern, s)
                term = [self.dict_words[word] for word in words]
                self.longest = max(self.longest, len(term))
                if i == 0:
                    self.train_s1_matrix.append(term)
                else:
                    self.train_s2_matrix.append(term)
        for item in self.test:
            for i in range(2):
                s = item[i]
                s = s.upper()
                words = re.findall(pattern, s)
                term = [self.dict_words[word] for word in words]
                self.longest = max(self.longest, len(term))
                if i == 0:
                    self.test_s1_matrix.append(term)
                else:
                    self.test_s2_matrix.append(term)
        self.len_words+=1
