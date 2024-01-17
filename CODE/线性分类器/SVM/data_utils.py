import pickle
import numpy as np
import os
#from scipy.misc import imread
from cv2 import imread

def load_CIFAR_batch(filename):
  """ 
  加载CIFAR数据集的一个批次
  参数：
    filename：批次数据文件名
  返回值：
    X：加载的数据，形状为(10000, 3, 32, 32)，数据类型为浮点型
    Y：加载的标签，形状为(10000,)，数据类型为NumPy数组
  """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='bytes')  # 使用pickle模块从文件中加载字节流数据
    X = datadict[b'data']  # 获取数据
    Y = datadict[b'labels']  # 获取标签
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")  # 对数据进行形状变换和类型转换
    Y = np.array(Y)  # 将标签转换为NumPy数组
    return X, Y  # 返回加载的数据和标签


def load_CIFAR10(ROOT):
  """ 
  加载所有 CIFAR 数据

  参数:
    ROOT: str, 数据集的根目录

  返回值:
    Xtr: 数组, 训练集的图像数据
    Ytr: 数组, 训练集的标签数据
    Xte: 数组, 测试集的图像数据
    Yte: 数组, 测试集的标签数据
  """
  xs = []  # 存储训练集的图像数据
  ys = []  # 存储训练集的标签数据
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))  # 数据文件路径
    X, Y = load_CIFAR_batch(f)  # 加载一个数据文件
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)  # 拼接训练集的图像数据
  Ytr = np.concatenate(ys)  # 拼接训练集的标签数据
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))  # 加载测试集数据
  return Xtr, Ytr, Xte, Yte  # 返回训练集和测试集的图像数据和标签数据


def load_tiny_imagenet(path, dtype=np.float32):
  """
  加载TinyImageNet数据集。每个 TinyImageNet-100-A，TinyImageNet-100-B 和
  TinyImageNet-200 有着相同的目录结构，所以可以用来加载任意一个数据集。

  输入:
  - path: 字符串，表示要加载的目录路径。
  - dtype: numpy 数据类型，表示加载数据时使用的数据类型。

  返回: 一个元组，其中包含
  - class_names: 列表，class_names[i] 是一个字符串列表，表示加载的数据集中类 i 的 WordNet 名称。
  - X_train: (N_tr, 3, 64, 64) 大小的数组，表示训练集图像数据。
  - y_train: (N_tr,) 大小的数组，表示训练集标签数据。
  - X_val: (N_val, 3, 64, 64) 大小的数组，表示验证集图像数据。
  - y_val: (N_val,) 大小的数组，表示验证集标签数据。
  - X_test: (N_test, 3, 64, 64) 大小的数组，表示测试集图像数据。
  - y_test: (N_test,) 大小的数组，表示测试集标签数据；如果测试集标签不可用（例如学生代码中），则 y_test 为 None。
  """
  # 首先加载 wnids
  with open(os.path.join(path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]

  # 将 wnids 映射为整数标签
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # 使用 words.txt 文件为每个类获取名称
  with open(os.path.join(path, 'words.txt'), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.iteritems():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]

  class_names = [wnid_to_words[wnid] for wnid in wnids]

  # 接下来加载训练数据
  X_train = []
  y_train = []
  for i, wnid in enumerate(wnids):
    if (i + 1) % 20 == 0:
      print('正在加载训练数据的 synset %d / %d' % (i + 1, len(wnids)))
    # 为了确定文件名，我们需要打开 boxes 文件
    boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      filenames = [x.split('\t')[0] for x in f]
    num_images = len(filenames)

    X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
    y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
    for j, img_file in enumerate(filenames):
      img_file = os.path.join(path, 'train', wnid, 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        img.shape = (64, 64, 1)
      X_train_block[j] = img.transpose(2, 0, 1)
    X_train.append(X_train_block)
    y_train.append(y_train_block)

  # 我们需要将所有的训练数据连接起来
  X_train = np.concatenate(X_train, axis=0)
  y_train = np.concatenate(y_train, axis=0)

  # 接下来加载验证数据
  with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
    img_files = []
    val_wnids = []
    for line in f:
      img_file, wnid = line.split('\t')[:2]
      img_files.append(img_file)
      val_wnids.append(wnid)
    num_val = len(img_files)
    y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
    X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
      img_file = os.path.join(path, 'val', 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        img.shape = (64, 64, 1)
      X_val[i] = img.transpose(2, 0, 1)

  # 接下来加载测试图像
  # 学生不会有测试标签，所以我们需要遍历图像文件夹中的所有文件。
  img_files = os.listdir(os.path.join(path, 'test', 'images'))
  X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
  for i, img_file in enumerate(img_files):
    img_file = os.path.join(path, 'test', 'images', img_file)
    img = imread(img_file)
    if img.ndim == 2:
      img.shape = (64, 64, 1)
    X_test[i] = img.transpose(2, 0, 1)

  y_test = None
  y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
  if os.path.isfile(y_test_file):
    with open(y_test_file, 'r') as f:
      img_file_to_wnid = {}
      for line in f:
        line = line.split('\t')
        img_file_to_wnid[line[0]] = line[1]
    y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
    y_test = np.array(y_test)

  return class_names, X_train, y_train, X_val, y_val, X_test, y_test


def load_models(models_dir):
    """
    从磁盘加载保存的模型。它会尝试反序列化目录中的所有文件，对于反序列化出错的文件（如README.txt）将会被跳过。

    参数：
    - models_dir: 包含模型文件的目录路径的字符串。每个模型文件都是带有'model'字段的pickle文件。

    返回值：
    - models: 模型文件名与模型之间的映射字典。
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = pickle.load(f)['model']
            except pickle.UnpicklingError:
                continue
    return models
