from feature_batch import Random_embedding, Glove_embedding, get_batch
import random
from plot_batch import NN_plot, NN_embedding
from neural_network import ESIM

if __name__ == '__main__':

    with open('snli_1.0_train.txt', 'r') as f: #'r'只读
        temp = f.readlines() #f.readlines()方法将整个文件读取到内存中，并以字符串列表的形式返回文件的每一行。

    with open('glove.6B.50d.txt', 'rb') as f: #‘rb’ 二进制读取
        lines = f.readlines()

    # 用GloVe创建词典
    trained_dict = dict()
    n = len(lines)
    for i in range(n):
        line = lines[i].split()
        trained_dict[line[0].decode("utf-8").upper()] = [float(line[j]) for j in range(1, 51)]

    data = temp[1:]

    learning_rate = 0.001
    len_feature = 50
    len_hidden = 50
    iter_times = 50
    batch_size = 1000

    # random embedding
    random.seed(2021)
    random_embedding = Random_embedding(data=data)
    random_embedding.get_words()
    random_embedding.get_id()

    # trained embedding : glove
    random.seed(2021)
    glove_embedding = Glove_embedding(data=data, trained_dict=trained_dict)
    glove_embedding.get_words()
    glove_embedding.get_id()

    NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, learning_rate, batch_size, iter_times)