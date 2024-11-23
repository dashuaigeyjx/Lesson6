# 深度学习基础
深度学习是机器学习的一个子集

区别：

1. 方法和模型复杂度：机器学习方法通常基于统计学和模式识别等原理，使用各种算法和模型来从数据中学习规律和模式。深度学习是机器学习的一个**子领域**，它专注于使用多层神经网络模型来学习更高级的抽象特征表示，可以处理更复杂的数据和任务。
2. 特征工程：在传统机器学习中，通常需要手动进行特征工程，即**从原始数据中提取和选择合适的特征**。而在深度学习中，神经网络可以**自动从原始数据中学习特征表示**，减少了对手动特征工程的依赖。
3. 数据需求：深度学习通常需要**大量的训练数据**来获得良好的性能，特别是在处理复杂任务时。机器学习方法相对而言**对数据量的要求较低**，但仍需要足够的数据进行模型训练。

联系：

1. 数据驱动：机器学习和深度学习都是数据驱动的方法，它们通过从数据中学习模式和规律来进行预测、分类或其他任务。
2. 模型训练：无论是机器学习还是深度学习，模型训练都是通过优化目标函数来调整模型参数，以最小化预测结果与真实标签之间的误差。这涉及到使用训练数据进行模型优化的过程。
3. 应用领域：机器学习和深度学习都在各种领域得到广泛应用，如计算机视觉、自然语言处理、语音识别、推荐系统等。它们可以用于解决分类、回归、聚类、生成等各种机器学习任务。
4. 结合使用：机器学习和深度学习并非相互排斥，而是可以结合使用。深度学习可以作为机器学习的一种工具或方法之一，用于处理特别复杂的任务，而机器学习方法可以用于处理数据较少或较简单的情况。

过程：

![img](https://ai-studio-static-online.cdn.bcebos.com/9f7cc7174c6f482b9b0d3a1f9bdc1195cf9bf0bc24d140da87aceba2dde4ea5d)

# 理论

> 喜欢看理论，喜欢数学公式推导的看这本书：https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#id5

这里是一些需要了解的名词：

- 数据集
> 数据集是一组相关的数据的集合，用于机器学习和数据分析任务。它可以包含各种类型的数据，如数字、文本、图像、音频等。数据集通常被分为训练集、测试集和验证集等不同的子集，用于不同的目的。
- 训练集
> 训练集是用于训练机器学习模型的数据子集。模型通过在训练集上学习数据中的模式和规律，调整自身的参数以最小化损失函数。训练集通常占数据集的较大比例，例如 70% - 80%。
- 测试集
> 测试集是用于评估机器学习模型性能的数据子集。在模型训练完成后，使用测试集来检验模型在新的、未见过的数据上的预测能力。测试集应该与训练集完全独立，以确保评估的准确性
- 标签
> 在监督学习中，标签是与输入数据对应的正确输出值。例如，在图像分类任务中，标签可以是图像所属的类别。模型通过学习输入数据与标签之间的关系，来进行预测
- 特征
> 特征是描述数据的属性或变量。在机器学习中，特征通常是从原始数据中提取出来的，用于表示数据的特定方面。例如，在图像识别中，特征可以是颜色、纹理、形状等；在文本分类中，特征可以是单词、短语、词性等。
- 权重、偏置
> 在神经网络中，权重和偏置是模型的参数。权重表示神经元之间连接的强度，偏置则是一个常数，用于调整神经元的输出。模型通过调整权重和偏置来最小化损失函数，从而学习数据中的模式和规律。
- 损失函数
> 损失函数是用于衡量模型预测值与真实值之间差异的函数。在训练机器学习模型时，目标是最小化损失函数。常见的损失函数有均方误差、交叉熵等。不同的任务和模型可能需要选择不同的损失函数。
- 梯度下降
> 梯度下降是一种优化算法，用于最小化损失函数。它通过计算损失函数对模型参数的梯度，然后沿着梯度的反方向更新参数，以逐步减小损失函数的值。梯度下降有多种变体，如随机梯度下降、批量梯度下降等。
- 优化算法
> 优化算法是用于调整模型参数以最小化损失函数的方法。除了梯度下降，还有许多其他的优化算法，如 Adam、Adagrad、RMSprop 等。这些算法在计算梯度和更新参数的方式上有所不同，具有不同的优缺点和适用场景
- 神经网络
- ...

## 多层感知机

- 大部分问题不是线性的，所以添加一个隐藏层，更易于捕捉到输入与输出之间复杂的相互作用，表示更复杂的模型
- 隐藏层也可以不只一个，可以用更深的网络，
- 可能更易于逼近一些函数

![img](["C:\Users\lenovo\Downloads\Courseware-Backend-Python-2023-main\Courseware-Backend-Python-2023-main\lesson-04\img\muti.png"](https://www.bing.com/images/search?view=detailV2&ccid=aqOKipo9&id=121C397999EFB9F83943E405926D98C1C858DC43&thid=OIP.aqOKipo9A7sx8zHvVbZFuwHaEL&mediaurl=https%3A%2F%2Fimg-blog.csdnimg.cn%2F20200517191914326.png&cdnurl=https%3A%2F%2Fth.bing.com%2Fth%2Fid%2FR.6aa38a8a9a3d03bb31f331ef55b645bb%3Frik%3DQ9xYyMGYbZIF5A%26pid%3DImgRaw%26r%3D0&exph=1110&expw=1968&q=%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA%E5%9B%BE%E7%89%87&simid=608034436525080492&FORM=IRPRST&ck=90C1AA6BDA7BDF0DE17986EE963553C6&selectedIndex=0&itb=0&cw=1390&ch=764&ajaxhist=0&ajaxserp=0))

### 激活函数

*激活函数*（activation function）通过计算加权和并加上偏置来确定神经元是否应该被激活， 它们将输入信号转换为输出的可微运算。

> 实际上，每一层的输入都可以用激活函数，但是要根据情况去选择。

#### ReLU函数

ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素。 

使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。

![image-20231122103605746](./img/relu.png)

![../_images/output_mlp_76f463_18_1.svg](https://zh.d2l.ai/_images/output_mlp_76f463_18_1.svg)

#### Sigmoid函数

*sigmoid函数*将输入变换为区间(0, 1)上的输出。 因此，sigmoid通常称为*挤压函数*（squashing function）： 它将范围（-inf, inf）中的任意输入压缩到区间（0, 1）中的某个值：

它是一个平滑的、可微的阈值单元近似。 当我们想要将输出视作二元分类问题的概率时， sigmoid仍然被广泛用作**输出**单元上的激活函数 （sigmoid可以视为softmax的特例）。 

![image-20231122104751861](./img/sigmoid.png)

![../_images/output_mlp_76f463_48_0.svg](https://zh.d2l.ai/_images/output_mlp_76f463_48_0.svg)

## 梯度下降

![深度学习实战教程(二)：线性单元和梯度下降](https://cuijiahua.com/wp-content/uploads/2018/11/dl-8-3.png)

### 随机梯度下降

![深度学习实战教程(二)：线性单元和梯度下降](https://cuijiahua.com/wp-content/uploads/2018/11/dl-8-4.png)


## 深度学习框架

深度学习框架是用于构建、训练和部署深度学习模型的软件工具集合。它们提供了一种方便的方式来定义、优化和执行神经网络模型。以下是一些常见的深度学习框架：

1. TensorFlow：由Google开发的最受欢迎的开源深度学习框架之一。它具有强大的灵活性和高性能，并支持在多种平台上进行部署，包括移动设备和大规模分布式系统。

   https://www.tensorflow.org/guide/eager?hl=zh-cn

2. PyTorch：由Facebook开发的深度学习框架，它提供了动态图计算的能力，使得模型的定义和调试变得更加直观。PyTorch还被广泛用于研究领域，因为它具有良好的可扩展性和易用性。

   https://pytorch.org/tutorials/

3. Keras：Keras是一个高级神经网络API，它**可以运行在多个深度学习框架**上，**包括TensorFlow、PyTorch**等。Keras的设计目标是提供简单易用的接口，使得快速原型设计和实验变得更加容易。

还有一个框架是paddlepaddle：https://www.paddlepaddle.org.cn/tutorials/projectdetail/5603475

> 我怀疑以上三个在我们今后的学习中都会用到，为什么呢？
>
> 因为你想在网上抄别人的模型来运行，他大概率用的是tensorflow或者pytorch，你要是不会的话可能都抄不明白。

我们的学习几乎要基于example，也就是不用自己写完整代码，只要能看懂，会改代码就行了。

接下来我们感受一下，顺便看看代码：

## 深度学习实战

### 线性回归

```python
# 基本的线性回归拟合线段
import torch
from torch.nn import Linear, MSELoss
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt

# print(torch.__version__)

x = np.linspace(0, 20, 500)
y = 5 * x + 7
plt.plot(x, y)
plt.show()

# 生成一些随机的点，来作为训练数据
x = np.random.rand(256)
noise = np.random.randn(256) / 4
y = x * 5 + 7 + noise

# 散点图
plt.scatter(x, y)
plt.show()

# 其中参数(1, 1)代表输入输出的特征(feature)数量都是1. Linear 模型的表达式是y=w⋅x+b其中 w代表权重， b代表偏置
model = Linear(1, 1)

# 损失函数我们使用均方损失函数：MSELoss
criterion = MSELoss()

# 优化器我们选择最常见的优化方法 SGD，就是每一次迭代计算 mini-batch 的梯度，然后对参数进行更新，学习率 0.01
optim = SGD(model.parameters(), lr=0.01)

# 训练3000次
epochs = 3000

# 准备训练数据: x_train, y_train 的形状是 (256, 1)，
# 代表 mini-batch 大小为256，
# feature 为1. astype('float32') 是为了下一步可以直接转换为 torch.float
x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

for i in range(epochs):
    # 整理输入和输出的数据，这里输入和输出一定要是torch的Tensor类型
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    # 使用模型进行预测
    outputs = model(inputs)
    # 梯度置0，否则会累加
    optim.zero_grad()
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 使用优化器默认方法优化
    optim.step()
    if (i % 100 == 0):
        # 每 100次打印一下损失函数，看看效果
        print('epoch {}, loss {:1.4f}'.format(i, loss.data.item()))

[w, b] = model.parameters()
print(w.item(), b.item())

predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'go', label='data', alpha=0.3)
plt.plot(x_train, predicted, label='predicted', alpha=1)
plt.legend()
plt.show()

x_new = np.array([[3.]], dtype=np.float32)
x_new_tensor = torch.from_numpy(x_new)
predicted_new = model(x_new_tensor).item()
print(predicted_new)
```

### 图像分类

pytorch官网的quick start , 先尝试自己运行

#### Pytorch

```bash
pip install torch # 或者
```

```bash
conda install torch # anaconda环境下
```

**代码**

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 下载训练数据和测试数据
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# 加载数据
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 获取第一批次的图像和标签
images, labels = next(iter(test_dataloader))
# 获取第一张图像
image = images[0]
label = labels[0]
# 将图像的形状从 [C, H, W] 转换为 [H, W, C]
image = image.permute(1, 2, 0)
# 将标签转换为对应的类别名称
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
class_name = class_names[label]
# 显示图像
plt.imshow(image)
plt.title(class_name)
plt.axis('off')
plt.show()

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # 打印输入数据 X 的形状
    print(f"Shape of y: {y.shape} {y.dtype}")  # 打印标签数据 y 的形状和数据类型
    break  # 仅打印一次后退出循环

# 获取用于训练的CPU、GPU或MPS设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        神经网络模型的初始化函数。

        参数：
            无输入参数。

        输出：
            无输出，用于初始化神经网络模型的各个层。

        """
        super().__init__()

        # 将输入数据展平
        self.flatten = nn.Flatten()

        # 定义线性层和激活函数的堆叠
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # 输入大小为 28 * 28，输出大小为 512
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(512, 512),  # 输入大小为 512，输出大小为 512
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(512, 10)  # 输入大小为 512，输出大小为 10
        )

    def forward(self, x):
        """
        神经网络模型的前向传播函数。

        参数：
            x (torch.Tensor): 输入数据。

        输出：
            logits (torch.Tensor): 模型的预测结果（未经过激活函数）。

        """
        # 展平输入数据
        x = self.flatten(x)

        # 通过线性层和激活函数的堆叠进行前向传播
        logits = self.linear_relu_stack(x)

        return logits


model = NeuralNetwork().to(device)  # 创建神经网络模型实例，并将其移动到指定的设备上（如 CPU 或 GPU）
print(model)  # 打印神经网络模型的结构

loss_fn = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 定义优化器为随机梯度下降（SGD），学习率为 0.001


def train(dataloader, model, loss_fn, optimizer):
    """
    对给定的数据加载器进行训练，更新模型的参数。

    参数：
        dataloader (torch.utils.data.DataLoader): 数据加载器，用于加载训练数据集。
        model (torch.nn.Module): 神经网络模型。
        loss_fn (torch.nn.Module): 损失函数，用于计算预测结果与真实标签之间的损失。
        optimizer (torch.optim.Optimizer): 优化器，用于更新模型的参数。

    输出：
        无返回值，用于训练和更新模型。

    """
    size = len(dataloader.dataset)  # 数据集的大小
    model.train()  # 设置模型为训练模式
    for batch, (X, y) in enumerate(dataloader):  # 遍历数据加载器中的每个批次
        X, y = X.to(device), y.to(device)  # 将输入数据和标签移动到指定的设备上（如 CPU 或 GPU）

        # 计算预测误差
        pred = model(X)  # 前向传播，获取模型的预测结果
        loss = loss_fn(pred, y)  # 计算预测结果与真实标签之间的损失

        # 反向传播和优化
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新模型的参数
        optimizer.zero_grad()  # 清空梯度，准备处理下一个批次的数据。

        if batch % 100 == 0:  # 如果当前批次是第 100 的倍数（用于控制打印频率）
            loss, current = loss.item(), (batch + 1) * len(X)  # 获取当前批次的损失值和已处理的样本数。
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # 打印当前批次的损失值和已处理的样本数。


def test(dataloader, model, loss_fn):
    """
    对给定的数据加载器进行测试，评估模型的性能。

    参数：
        dataloader (torch.utils.data.DataLoader): 数据加载器，用于加载测试数据集。
        model (torch.nn.Module): 神经网络模型。
        loss_fn (torch.nn.Module): 损失函数，用于计算预测结果与真实标签之间的损失。

    输出：
        无返回值，打印测试结果。

    """
    size = len(dataloader.dataset)  # 数据集的大小
    num_batches = len(dataloader)  # 批次的数量
    model.eval()  # 设置模型为评估模式，这将禁用一些特定于训练的操作，如 Dropout。
    test_loss, correct = 0, 0  # 初始化测试损失和正确预测的数量为0
    with torch.no_grad():  # 在评估模式下，不需要计算梯度，因此使用 torch.no_grad() 上下文管理器来加速运算。
        for X, y in dataloader:  # 遍历数据加载器中的每个批次
            X, y = X.to(device), y.to(device)  # 将输入数据和标签移动到指定的设备上（如 CPU 或 GPU）
            pred = model(X)  # 前向传播，获取模型的预测结果
            test_loss += loss_fn(pred, y).item()  # 累加当前批次的损失值
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 累加正确预测的数量
    test_loss /= num_batches  # 计算平均测试损失
    correct /= size  # 计算准确率
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 训练5轮，可以调整
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

#### Tensorflow

这个是tensorflow给的示例代码，主要在用其中的keras，先尝试自己运行起来。

**安装**

```bash
pip install tensorflow # 或者
```

```bash
conda install tensorflow # anaconda环境下
```

默认是2.14版本，官方文档也是基于2.14版本的，如果你在网上查教程，那很有可能运行不了。（所以如果在github上找别人的代码也可能要尝试回退版本才能运行）

这就突出了anaconda的好处，你可以在不同环境上装不同版本的，就不用每次pip来回调了（但同时也说明不安装anaconda也能正常“玩”）

```bash
pip install tensorflow == <版本号> # 回退版本
```

```bash
conda install tensorflow == <版本号> # anaconda环境下
```

##### 训练

```python
# TensorFlow and tf.keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# print(tf.__version__)

# 加载 Fashion MNIST 数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
# 将数据集拆分为训练集和测试集
# 训练集包含用于训练模型的图像和标签
# 测试集包含用于评估模型性能的图像和标签
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 存储类名
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 训练集中有 60,000 个图像，每个图像由 28 x 28 的像素表示
print(train_images.shape)
#  (60000, 28, 28)

# 训练集有60000个标签
print(train_labels)
print(len(train_labels))
# 60000

# 测试集中有 10,000 个图像。同样，每个图像都由 28x28 个像素
print(test_images.shape)
print(len(test_labels))

# 预处理
plt.figure()  # 创建一个新的图形（图表）窗口
plt.imshow(train_images[9])  # 在图表上显示train_images中索引为9的图像
plt.colorbar()  # 添加一个颜色条，用于显示图像中颜色对应的数值
plt.grid(False)  # 不显示网格线
plt.show()  # 显示图表窗口

# 将这些值缩小至 0 到 1 之间，然后将其馈送到神经网络模型。
train_images = train_images / 255.0

test_images = test_images / 255.0

# 为了验证数据的格式是否正确，以及您是否已准备好构建和训练网络，
# 让我们显示训练集中的前 25 个图像，并在每个图像下方显示类名称。
# 创建一个大小为 10x10 的图像窗口
plt.figure(figsize=(10, 10))

# 循环处理 25 个图像
for i in range(25):
    # 在 5x5 的子图中绘制当前图像
    plt.subplot(5, 5, i + 1)

    # 移除 x 轴和 y 轴的刻度
    plt.xticks([])
    plt.yticks([])

    # 关闭网格线
    plt.grid(False)

    # 显示当前训练集图像
    plt.imshow(train_images[i], cmap=plt.cm.binary)

    # 在图像下方显示标签类别名称
    plt.xlabel(class_names[train_labels[i]])

# 显示图像窗口
plt.show()

# 创建一个序列模型
model = tf.keras.Sequential([
    # 将输入数据展平为一维向量，输入形状为 (28, 28)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 添加一个具有 128 个神经元的全连接层，并使用 ReLU 激活函数
    tf.keras.layers.Dense(128, activation='relu'),
    # 添加一个具有 10 个神经元的全连接层，该层输出未经过激活的原始预测分数
    tf.keras.layers.Dense(10)
])

# 编译模型
model.compile(
    # 使用了Adam优化器，Adam是一种常用的优化算法，用于调整模型的权重以最小化损失函数。
    optimizer='adam',
    # 稀疏分类交叉熵损失函数。该损失函数适用于多类别分类问题，其中目标标签是整数形式（而不是独热编码）。
    # from_logits=True表示模型的输出是未经过概率分布转换的原始预测分数
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # 指定了评估模型性能的指标，即准确率。在训练过程中，模型将根据准确率指标来评估和监控其性能
    metrics=['accuracy'])

# 训练模型
# 向模型馈送数据
model.fit(train_images, train_labels, epochs=10)
model.save('image_classify.h5')
```

##### 测试

```python
# TensorFlow and tf.keras
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

matplotlib.use('TkAgg')
# Helper libraries

# print(tf.__version__)

# 加载 Fashion MNIST 数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
# 将数据集拆分为训练集和测试集
# 训练集包含用于训练模型的图像和标签
# 测试集包含用于评估模型性能的图像和标签
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 存储类名
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 获取训练好的模型
model = keras.models.load_model('image_classify.h5')
# 评估准确率
# 控制输出信息详细程度的参数。当设置为2时，会显示每个测试样本的评估进度条和相关的指标
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# 进行预测
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print(predictions[0])

# 最有可能的
print(np.argmax(predictions[0]))

# 答案
print(test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 验证预测结果
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 使用训练好的模型
# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

img = (np.expand_dims(img, 0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
```

# 作业

1. 跑通本节课所有代码，截图。
2. 用多层感知机训练一个 y=x1的平方+x2的平方 的函数（pytorch/tensorflow）
3. 有空自己看看pytorch官网的exmaple代码并运行，随便什么都行，运行的话可以截图提交

提交到 1748349252@qq.com 署名+学号
