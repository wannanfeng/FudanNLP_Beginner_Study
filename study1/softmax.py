import numpy
import random


class Softmax:
    """Softmax regression"""
    def __init__(self, sample, typenum, feature):
        self.sample = sample  # 训练集样本个数
        self.typenum = typenum  # （情感）种类个数
        self.feature = feature  # 0-1向量的长度:即[0,0,0,......1,1,0,......]长度
        self.W = numpy.random.randn(feature, typenum)  # 参数矩阵W初始化

    def softmax_calculation(self, x):
        """x是向量，计算softmax值"""
        exp = numpy.exp(x - numpy.max(x))  # 先减去最大值防止指数太大溢出
        return exp / exp.sum()

    def softmax_all(self, wtx):
        """wtx是矩阵，即许多向量叠在一起，按行计算softmax值"""
        wtx -= numpy.max(wtx, axis=1, keepdims=True) # 先减去行最大值防止指数太大溢出
        wtx = numpy.exp(wtx)
        wtx /= numpy.sum(wtx, axis=1, keepdims=True)
        return wtx

    def change_y(self, y):
        """把（情感）种类转换为一个one-hot向量"""
        ans = numpy.array([0] * self.typenum)  #生成为整数的  如果是np.zero生成是小数的
        ans[y] = 1
        return ans.reshape(-1, 1)

    def prediction(self, X):
        """给定0-1矩阵X，计算每个句子的y_hat值（概率）"""
        prob = self.softmax_all(X.dot(self.W))
        # 使用softmax函数对结果进行归一化处理得到各个分类的概率分布，最后返回概率最大的分类。
        return prob.argmax(axis=1)

    def correct_rate(self, train, train_y, test, test_y):
        """计算训练集和测试集的准确率"""
        # train set
        # 首先计算训练集和测试集上的预测结果 pred_train 和 pred_test。
        n_train = len(train)
        pred_train = self.prediction(train)
        # 计算预测结果与真实标签相同的样本数占总样本数的比例，即为分类准确率。
        train_correct = sum([train_y[i] == pred_train[i] for i in range(n_train)]) / n_train
        # test set
        n_test = len(test)
        pred_test = self.prediction(test)
        test_correct = sum([test_y[i] == pred_test[i] for i in range(n_test)]) / n_test
        print(train_correct, test_correct)
        return train_correct, test_correct

    # X：numpy 数组，表示输入数据，每行代表一个数据样本，每列代表一个特征。
    # y：numpy 数组，表示输出数据，它的每个元素与 X 中的一行对应。
    # alpha：学习率，用于指定每次更新参数时的步长。
    # times：训练次数，即迭代次数。
    # strategy：用于指定使用哪种训练策略，有 "mini"、"shuffle" 和 "batch" 三种可选，分别表示小批量梯度下降、随机梯度下降和批量梯度下降。
    def regression(self, X, y, alpha, times, strategy="mini", mini_size=100):
        """Softmax regression"""
        if self.sample != len(X) or self.sample != len(y):
            raise Exception("Sample size does not match!")  # 样本个数不匹配
        if strategy == "mini":  # mini-batch
            for i in range(times):
                increment = numpy.zeros((self.feature, self.typenum))  # 梯度初始为0矩阵
                for j in range(mini_size):  # 随机抽K次
                    k = random.randint(0, self.sample - 1)  #返回指定范围内的整数
                    yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                    # increment 变量用于存储每个小批量数据的梯度之和
                    # feature行1列*1行typenum列
                    increment += X[k].reshape(-1, 1).dot((self.change_y(y[k]) - yhat).T)  # 梯度加和
                # print(i * mini_size)
                self.W += alpha / mini_size * increment  # 参数更新
        elif strategy == "shuffle":  # 随机梯度
            for i in range(times):
                k = random.randint(0, self.sample - 1)  # 每次抽一个
                yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                # self.change_y(y[k]) - yhat表示预测值与真实值之间的误差
                increment = X[k].reshape(-1, 1).dot((self.change_y(y[k]) - yhat).T)  # 计算梯度
                self.W += alpha * increment  # 参数更新
                # if not (i % 10000):
                #     print(i)
        elif strategy=="batch":  # 整批量梯度
            for i in range(times):
                increment = numpy.zeros((self.feature, self.typenum))  ## 梯度初始为0矩阵
                for j in range(self.sample):  # 所有样本都要计算
                    yhat = self.softmax_calculation(self.W.T.dot(X[j].reshape(-1, 1)))
                    increment += X[j].reshape(-1, 1).dot((self.change_y(y[j]) - yhat).T)  # 梯度加和
                # print(i)
                self.W += alpha / self.sample * increment  # 参数更新
        else:
            raise Exception("Unknown strategy")
