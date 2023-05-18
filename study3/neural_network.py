import torch
import torch.nn as nn

#ESIM模型

class Input_Encoding(nn.Module):  #第一层
    def __init__(self, len_hidden, len_feature, len_words, longest, weight = None, layer = 1, batch_first = True, drop_out = 0.5):
        super().__init__()
        self.len_feature = len_feature
        self.len_words = len_words
        self.len_hidden = len_hidden
        self.layer = layer
        self.longest = longest
        self.drop_out = nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first, bidirectional=True).cuda()
    def forward(self, x):
        x = torch.LongTensor(x).cuda()
        x = self.embedding(x)
        x = self.drop_out(x)
        self.lstm.flatten_parameters()
        #权重和偏置参数展开(flatten)，以便进行更快的计算，并减少GPU内存使用,运行正向或反向传递之前，需要对每个LSTM层调用
        x, _ = self.lstm(x)
        return x

class Local_Inference_Modeling(nn.Module): #第二层
    def __init__(self):
        super().__init__()
        self.softmax1 = nn.Softmax(dim=1).cuda()
        self.softmax2 = nn.Softmax(dim=2).cuda()

    def forward(self, a_bar, b_bar):
        e = torch.matmul(a_bar, b_bar.transpose(1, 2)).cuda()  #sumdim=3 交换dim=1和dim=2的维度

        a_tilde = self.softmax2(e)
        a_tilde = a_tilde.bmm(b_bar)
        b_tilde = self.softmax1(e)
        b_tilde = b_tilde.transpose(1, 2).bmm(a_bar)

        m_a = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde], dim=-1)
        m_b = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde], dim=-1)

        return m_a, m_b

class Inference_Composition(nn.Module): #第三层

    def __init__(self, len_feature, len_hidden_m, len_hidden, layer=1, batch_first=True, drop_out=0.5):
        super().__init__()
        self.linear = nn.Linear(len_hidden_m, len_feature).cuda()
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first, bidirectional=True).cuda()
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.linear(x)
        x = self.drop_out(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x

class Prediction(nn.Module):
    def __init__(self, len_v, len_mid, type_num=4, drop_out=0.5):
        super().__init__()
        self.mlp = nn.Sequential(nn.Dropout(drop_out),nn.Linear(len_v, len_mid), nn.Tanh(), nn.Linear(len_mid, type_num)).cuda() #将模型的多个层组合在一起，以形成一个整体的神经网络模型
    def forward(self, a, b):
        v_a_average = a.sum(1)/a.shape[1]
        v_a_max = a.max(1)[0]

        v_b_average = b.sum(1)/b.shape[1]
        v_b_max = b.max(1)[0]
        output = torch.cat((v_a_average,v_a_max,v_b_average,v_b_max),dim=-1)
        return self.mlp(output)

class ESIM(nn.Module):  # 总
    def __init__(self, len_feature, len_hidden, len_words,longest, type_num=4, weight=None, layer=1, batch_first=True,
                 drop_out=0.5):
        super().__init__()
        self.len_words = len_words
        self.longest = longest
        self.input_encoding = Input_Encoding(len_feature, len_hidden, len_words,longest, weight=weight, layer=layer,
                                             batch_first=batch_first, drop_out=drop_out)
        self.local_inference_modeling = Local_Inference_Modeling()
        self.inference_composition = Inference_Composition(len_feature, 8 * len_hidden, len_hidden, layer=layer,
                                                           batch_first=batch_first, drop_out=drop_out)
        self.prediction = Prediction(len_hidden*8, len_hidden, type_num=type_num, drop_out=drop_out)

    def forward(self,a,b):
        a_bar=self.input_encoding(a)
        b_bar=self.input_encoding(b)

        m_a,m_b=self.local_inference_modeling(a_bar,b_bar)

        v_a=self.inference_composition(m_a)
        v_b=self.inference_composition(m_b)

        out_put=self.prediction(v_a,v_b)

        return out_put