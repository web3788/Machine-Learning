import numpy as np


class SVM(object):
    def __init__(self):
        # A numpy array of shape (C, D)
        self.W = None

    def train(self, X, y, reg, delta, learning_rate, batch_num, num_iter, output):
        """训练SVM模型

        :param X: 形状为(N, D)的numpy数组，表示训练样本的特征矩阵
        :param y: 形状为(N,)的numpy数组，表示训练样本的标签
        :param reg: 形状为(N,)的numpy数组，表示正则化参数
        :param delta: 范数的 margin
        :param learning_rate: 梯度下降的速率
        :param batch_num: 每步Mini-batch梯度下降使用的训练样本数量
        :param num_iter: 优化步骤的次数
        :return: 损失的历史记录
        """
        num_train = X.shape[0]  # 训练样本的数量
        num_dim = X.shape[1]  # 特征矩阵的维度
        num_classes = np.max(y) + 1  # y的取值为0...K-1，表示总共有K个类别

        if self.W is None:
            # 懒惰初始化W
            self.W = 0.001 * np.random.randn(num_classes, num_dim)


        loss_history = []
        for i in range(num_iter):
            # 随机选择Mini-batch
            sample_index = np.random.choice(num_train, batch_num, replace=False)
            X_batch = X[sample_index, :]
            y_batch = y[sample_index]


            loss, gred = self.svm_cost_function(X_batch, y_batch, reg, delta)
            loss_history.append(loss)
            

            self.W -= learning_rate * gred

            if output and  i % 100 == 0:
                    print('Iteration %d / %d: loss %f' % (i, num_iter, loss))

        return loss_history

    def predict(self, X):
        """预测函数

        :param X: 形状为 (N, D) 的numpy数组
        :return: y_pred, 预测结果的numpy数组，形状为 (N, )
        """
        scores = X.dot(self.W.T)

        y_pred = np.zeros(X.shape[0])
        y_pred = np.argmax(scores, axis=1)

        return y_pred

    def svm_cost_function(self, X, y, reg, delta):
        """计算损失

        :param X: 形状为(N, D)的numpy数组，表示特征矩阵
        :param y: 形状为(N,)的numpy数组，表示标签数组
        :param reg: 正则化强度
        :param delta: 范数
        :return: 损失，梯度
        """
        num_train = X.shape[0]

        scores = X.dot(self.W.T)  # N * C
        correct_class_scores = scores[range(num_train), y]
        margins = scores - correct_class_scores[:, np.newaxis] + delta
        margins = np.maximum(0, margins)
        # 不要忽略它，因为 'y - y + delta' > 0，我们应该将其重置为0
        margins[range(num_train), y] = 0

        loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(self.W * self.W)

        # 计算梯度 [对于每个例子，当margin > 0时，正确的标签的W应该为-X，错误的标签的W应该为+X]
        ground_true = np.zeros(margins.shape)  # N * C
        ground_true[margins > 0] = 1
        sum_margins = np.sum(ground_true, axis=1)
        ground_true[range(num_train), y] -= sum_margins

        gred = ground_true.T.dot(X) / num_train + reg * self.W

        return loss, gred