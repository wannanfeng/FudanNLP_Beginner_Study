import sys
import numpy
import csv
import random
from feature import Bag,Gram
from plotshow import alpha_gradient_plot

# 数据读取
with open('train.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)
# print(len(temp))  # 156061
# 初始化
data = temp[1:]
max_item=1000  #一次性选择1000个数据进行 mini-batch???
random.seed(2021)
numpy.random.seed(2021)

# 特征提取
bag=Bag(data,max_item)
#获取输入数据中出现的所有单词
bag.get_words()
# 将输入数据转换成基于词袋模型的特征矩阵
bag.get_matrix()
# print(bag.len) #363

gram=Gram(data, dimension=2, max_item=max_item)
gram.get_words()
gram.get_matrix()

# 画图

alpha_gradient_plot(bag,gram,10000,10)  # 计算10000次
# alpha_gradient_plot(bag,gram,100000,10)  # 计算100000次
