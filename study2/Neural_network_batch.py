import torch
import torch.nn as nn
import torch.nn.functional as F


class MY_RNN(nn.Module):
    """自己设计的RNN网络"""


    def __init__(self, len_feature, len_hidden, len_words, typenum=5, weight=None, layer=1, nonlinearity='tanh',
                 batch_first=True, drop_out=0.5):
        super(MY_RNN, self).__init__()
        self.len_feature = len_feature  # d的大小
        self.len_hidden = len_hidden  # l_h的大小
        self.len_words = len_words  # 单词的个数（包括padding）
        self.layer = layer  # 隐藏层层数
        self.dropout = nn.Dropout(drop_out)  # dropout层
        if weight is None:  # 随机初始化
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:  # GloVe初始化
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        # 用nn.Module的内置函数定义隐藏层
        self.rnn = nn.RNN(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, nonlinearity=nonlinearity,
                          batch_first=batch_first, dropout=drop_out).cuda()
        # 全连接层
        self.fc = nn.Linear(len_hidden, typenum).cuda()
        # 冗余的softmax层，可以不加
        # self.act = nn.Softmax(dim=1)


    def forward(self, x):
        """x:数据，维度为[batch_size， 句子长度]"""
        x = torch.LongTensor(x).cuda()
        batch_size = x.size(0)
        """经过词嵌入后，维度为[batch_size，句子长度，d]"""
        out_put = self.embedding(x)  # 词嵌入
        out_put = self.dropout(out_put)  # dropout层

        # 另一种初始化h_0的方式
        # h0 = torch.randn(self.layer, batch_size, self.len_hidden).cuda()
        # 初始化h_0为0向量
        h0 = torch.autograd.Variable(torch.zeros(self.layer, batch_size, self.len_hidden)).cuda()
        """dropout后不变，经过隐藏层后，维度为[1，batch_size, l_h]"""
        _, hn = self.rnn(out_put, h0)  # 隐藏层计算
        """经过全连接层后，维度为[1，batch_size, 5]"""
        out_put = self.fc(hn).squeeze(0)  # 全连接层
        """挤掉第0维度，返回[batch_size, 5]的数据"""
        # out_put = self.act(out_put)  # 冗余的softmax层，可以不加
        return out_put


class MY_CNN(nn.Module):
    def __init__(self, len_feature, len_words, longest, typenum=5, weight=None, drop_out=0.5):
        super(MY_CNN, self).__init__()
        self.len_feature = len_feature  # d的大小
        self.len_words = len_words  # 单词数目
        self.longest = longest  # 最长句子单词书目
        self.dropout = nn.Dropout(drop_out)  # Dropout层
        if weight is None:  # 随机初始化
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:  # GloVe初始化
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        # Conv2d参数详解：（输入通道数：1，输出通道数：l_l，卷积核大小：（行数，列数））
        # padding是指往句子两侧加 0，因为有的句子只有一个单词
        # 那么 X 就是 1*50 对 W=2*50 的卷积核根本无法进行卷积操作
        # 因此要在X两侧行加0（两侧列不加），（padding=（1，0））变成 3*50
        # 又比如 padding=（2，0）变成 5*50
        self.conv1 = nn.Sequential(nn.Conv2d(1, longest, (2, len_feature), padding=(1, 0)),
                                   nn.ReLU()).cuda()  # 第1个卷积核+激活层
        self.conv2 = nn.Sequential(nn.Conv2d(1, longest, (3, len_feature), padding=(1, 0)),
                                   nn.ReLU()).cuda()  # 第2个卷积核+激活层
        self.conv3 = nn.Sequential(nn.Conv2d(1, longest, (4, len_feature), padding=(2, 0)),
                                   nn.ReLU()).cuda()  # 第3个卷积核+激活层
        self.conv4 = nn.Sequential(nn.Conv2d(1, longest, (5, len_feature), padding=(2, 0)),
                                   nn.ReLU()).cuda()  # 第4个卷积核+激活层
        # 全连接层
        self.fc = nn.Linear(4 * longest, typenum).cuda()
        # 冗余的softmax层，可以不加
        # self.act = nn.Softmax(dim=1)

    def forward(self, x):
        """x:数据，维度为[batch_size， 句子长度]"""

        x = torch.LongTensor(x).cuda()
        """经过词嵌入后，维度为[batch_size，1，句子长度，d]"""
        out_put = self.embedding(x).view(x.shape[0], 1, x.shape[1], self.len_feature)  # 词嵌入
        """dropout后不变,记为X"""
        out_put = self.dropout(out_put)  # dropout层

        """X经过2*d卷积后，维度为[batch_size，l_l，句子长度+2-1，1]"""

        """挤掉第三维度（维度从0开始），[batch_size，l_l，句子长度+2-1]记为Y_1"""
        """注意：句子长度+2-1的2是padding造成的行数扩张"""
        conv1 = self.conv1(out_put).squeeze(3)  # 第1个卷积

        """X经过3*d卷积后，维度为[batch_size，l_l，句子长度+2-2，1]"""


        """挤掉第三维度（维度从0开始），[batch_size，l_l，句子长度+2-2]记为Y_2"""
        conv2 = self.conv2(out_put).squeeze(3)  # 第2个卷积

        """X经过4*d卷积后，维度为[batch_size，l_l，句子长度+4-3，1]"""
        """挤掉第三维度（维度从0开始），[batch_size，l_l，句子长度+4-3]记为Y_3"""
        conv3 = self.conv3(out_put).squeeze(3)  # 第3个卷积

        """X经过5*d卷积后，维度为[batch_size，l_l，句子长度+4-4，1]"""
        """挤掉第三维度（维度从0开始），[batch_size，l_l，句子长度+4-4]记为Y_4"""
        conv4 = self.conv4(out_put).squeeze(3)  # 第4个卷积

        """分别对（Y_1,Y_2,Y_3,Y_4）的第二维（维度从0开始）进行pooling"""
        """得到4个[batch_size,，l_l，1]的向量"""
        pool1 = F.max_pool1d(conv1, conv1.shape[2])
        pool2 = F.max_pool1d(conv2, conv2.shape[2])
        pool3 = F.max_pool1d(conv3, conv3.shape[2])
        pool4 = F.max_pool1d(conv4, conv4.shape[2])

        """拼接得到[batch_size,，l_l*4，1]的向量"""
        """挤掉第二维（维度从0开始）为[batch_size,，l_l*4]"""
        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2)  # 拼接起来
        """经过全连接层后，维度为[batch_size, 5]"""
        out_put = self.fc(pool)  # 全连接层
        # out_put = self.act(out_put)  # 冗余的softmax层，可以不加
        return out_put

