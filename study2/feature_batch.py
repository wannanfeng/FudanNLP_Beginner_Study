import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


def data_split(data, test_rate=0.3):
    """把数据按一定比例划分成训练集和测试集"""
    train = list()
    test = list()
    for datum in data:
        if random.random() > test_rate:
            train.append(datum)
        else:
            test.append(datum)
    return train, test


class Random_embedding():

    def __init__(self, data, test_rate=0.3):
        self.dict_words = dict()  # 单词->ID的映射
        data.sort(key=lambda x:len(x[2].split()))  # 按照句子长度排序，短着在前，这样做可以避免后面一个batch内句子长短不一，导致padding过度
        self.data = data
        self.len_words = 0  # 单词数目（包括padding的ID：0）
        self.train, self.test = data_split(data, test_rate=test_rate)  # 训练集测试集划分
        self.train_y = [int(term[3]) for term in self.train]  # 训练集类别
        self.test_y = [int(term[3]) for term in self.test]  # 测试集类别
        self.train_matrix = list()  # 训练集的单词ID列表，叠成一个矩阵
        self.test_matrix = list()  # 测试集的单词ID列表，叠成一个矩阵
        self.longest=0  # 记录最长的单词

    def get_words(self):
        for term in self.data:
            s = term[2]  # 取出句子
            s = s.upper()  # 记得要全部转化为大写！！（或者全部小写，否则一个单词例如i，I会识别成不同的两个单词）
            words = s.split()
            for word in words:  # 一个一个单词寻找
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)+1  # padding是第0个，所以要+1
        self.len_words=len(self.dict_words)  # 单词数目（暂未包括padding的ID：0）

    def get_id(self):
        for term in self.train:  # 训练集
            s = term[2]
            s = s.upper()
            words = s.split()
            item=[self.dict_words[word] for word in words]  # 找到id列表（未进行padding）
            self.longest=max(self.longest,len(item))  # 记录最长的单词
            self.train_matrix.append(item)
        for term in self.test:
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]  # 找到id列表（未进行padding）
            self.longest = max(self.longest, len(item))  # 记录最长的单词
            self.test_matrix.append(item)
        self.len_words += 1   # 单词数目（包括padding的ID：0）


class Glove_embedding():
    def __init__(self, data,trained_dict,test_rate=0.3):
        self.dict_words = dict()  # 单词->ID的映射
        self.trained_dict=trained_dict  # 记录预训练词向量模型
        data.sort(key=lambda x:len(x[2].split()))  # 按照句子长度排序，短着在前，这样做可以避免后面一个batch内句子长短不一，导致padding过度
        self.data = data
        self.len_words = 0  # 单词数目（包括padding的ID：0）
        self.train, self.test = data_split(data, test_rate=test_rate)  # 训练集测试集划分
        self.train_y = [int(term[3]) for term in self.train]  # 训练集类别
        self.test_y = [int(term[3]) for term in self.test]  # 测试集类别
        self.train_matrix = list()  # 训练集的单词ID列表，叠成一个矩阵
        self.test_matrix = list()  # 测试集的单词ID列表，叠成一个矩阵
        self.longest=0  # 记录最长的单词
        self.embedding=list()  # 抽取出用到的（预训练模型的）单词

    def get_words(self):
        self.embedding.append([0] * 50)  # 先加padding的词向量
        for term in self.data:
            s = term[2]  # 取出句子
            s = s.upper()  # 记得要全部转化为大写！！（或者全部小写，否则一个单词例如i，I会识别成不同的两个单词）
            words = s.split()
            for word in words:  # 一个一个单词寻找
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)+1  # padding是第0个，所以要+1
                    if word in self.trained_dict:  # 如果预训练模型有这个单词，直接记录词向量
                        self.embedding.append(self.trained_dict[word])
                    else:  # 预训练模型没有这个单词，初始化该词对应的词向量为0向量
                        # print(word)
                        # raise Exception("words not found!")
                        self.embedding.append([0]*50)
        self.len_words=len(self.dict_words)  # 单词数目（暂未包括padding的ID：0）

    def get_id(self):
        for term in self.train:  # 训练集
            s = term[2]
            s = s.upper()
            words = s.split()
            item=[self.dict_words[word] for word in words]  # 找到id列表（未进行padding）
            self.longest=max(self.longest,len(item))  # 记录最长的单词
            self.train_matrix.append(item)
        for term in self.test:
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]  # 找到id列表（未进行padding）
            self.longest = max(self.longest, len(item))  # 记录最长的单词
            self.test_matrix.append(item)
        self.len_words += 1  # 单词数目（暂未包括padding的ID：0）


class ClsDataset(Dataset):

    def __init__(self, sentence, emotion):
        self.sentence = sentence  # 句子
        self.emotion= emotion  # 情感类别

    def __getitem__(self, item):
        return self.sentence[item], self.emotion[item]

    def __len__(self):
        return len(self.emotion)


def collate_fn(batch_data):

    sentence, emotion = zip(*batch_data)
    sentences = [torch.LongTensor(sent) for sent in sentence]  # 把句子变成Longtensor类型
    padded_sents = pad_sequence(sentences, batch_first=True, padding_value=0)  # 自动padding操作！！！
    return torch.LongTensor(padded_sents), torch.LongTensor(emotion)


def get_batch(x,y,batch_size):

    dataset = ClsDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True,collate_fn=collate_fn)
    #  shuffle是指每个epoch都随机打乱数据排列再分batch，
    #  这里一定要设置成false，否则之前的排序会直接被打乱，
    #  drop_last是指不利用最后一个不完整的batch（数据大小不能被batch_size整除）
    return dataloader

