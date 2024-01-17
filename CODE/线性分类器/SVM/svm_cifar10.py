import numpy as np
from data_utils import load_CIFAR10
from svm import SVM
import matplotlib.pyplot as plt


def VisualizeImage(X_train, y_train):

    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 8
    for y, cls in enumerate(classes):
        # 得到该标签训练样本下标索引
        idxs = np.flatnonzero(y_train == y)
        # 从某一分类的下标中随机选择8个图像（replace设为False确保不会选择到同一个图像）
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        # 将每个分类的8个图像显示出来
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            # 创建子图像
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            # 增加标题
            if i == 0:
                plt.title(cls)
    plt.show()


def VisualizeLoss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()


def pre_dataset():
    # 加载CIFAR-10数据集
    cifar10_dir = 'D:\Python38\WORKS\CODE\svm\cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # 可视化训练集图像
    VisualizeImage(X_train, y_train)
    input('按下任意键进行交叉验证...')
    
    # 验证集划分
    num_train = 49000
    num_val = 1000
    sample_index = range(num_train, num_train + num_val)
    X_val = X_train[sample_index]
    y_val = y_train[sample_index]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]
    
    # 减去均值图像
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    mean_image = np.mean(X_train, axis=0)
    X_train = X_train - mean_image
    X_test = X_test - mean_image
    X_val = X_val - mean_image
    
    # 添加参数W
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    
    return X_train, y_train, X_test, y_test, X_val, y_val


def auto_get_parameter(X_train, y_train, X_val, y_val):
    # 学习率
    learning_rates = [1e-7, 5e-5]
    # 正则化强度
    regularization_strengths = [5e4, 1e5]

    # 初始化最佳参数为None，最佳验证准确率为负无穷
    best_parameter = None
    best_val = -1

    # 遍历所有学习率和正则化强度的组合
    for i in learning_rates:
        for j in regularization_strengths:
            # 创建SVM对象
            svm = SVM()
            # 调用SVM的训练方法，传入训练数据和参数进行训练
            svm.train(X_train, y_train, j, 1, i, 200, 1500, True)
            # 使用训练好的模型对验证集进行预测
            y_pred = svm.predict(X_val)
            # 计算预测准确率
            acc_val = np.mean(y_val == y_pred)
            # 若当前的验证准确率比之前的最佳验证准确率高，则更新最佳验证准确率和最佳参数
            if best_val < acc_val:
                best_val = acc_val
                best_parameter = (i, j)

    # 打印最佳参数和最佳验证准确率
    print('OK! 已确定参数！交叉验证中达到的最佳验证准确率为：%f' % best_val)
    return best_parameter


def get_svm_model(parameter, X, y):
    """
    获取SVM模型

    参数：
    parameter (tuple): 参数元组，包含SVM的核函数类型和核函数参数
    X (array-like, shape (n_samples, n_features)): 训练样本特征矩阵
    y (array-like, shape (n_samples,)): 训练样本标签向量

    返回：
    SVM: 训练好的SVM模型对象
    """
    svm = SVM()
    loss_history = svm.train(X, y, parameter[1], 1, parameter[0], 200, 1500, True)
    VisualizeLoss(loss_history)
    input('Enter any key to predict...')
    return svm


if __name__ == '__main__':
    # 数据预处理
    X_train, y_train, X_test, y_test, X_val, y_val = pre_dataset()
    # 参数确定
    best_parameter = auto_get_parameter(X_train, y_train, X_val, y_val)
    # 构建SVM模型
    svm = get_svm_model(best_parameter, X_train, y_train)
    # 准确率
    y_pred = svm.predict(X_test)
    print('Accuracy achieved during cross-validation: %f' % (np.mean(y_pred == y_test)))