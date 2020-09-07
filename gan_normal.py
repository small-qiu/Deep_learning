# coding=utf-8
import argparse   # 解析命令行参数和选项
import numpy as np   # numpy科学计算的库，可以提供矩阵运算
from scipy.stats import norm   # scipy数值计算库
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation  # matplotlib绘图库
import seaborn as sns  # 数据模块可视化

sns.set(color_codes=True)  # sns.set(style="white", palette="muted", color_codes=True)
# #set( )设置主题，调色板更常用 ,muted,柔和的

seed = 42  # 设置seed，使得每次生成的随机数相同
np.random.seed(seed)
tf.set_random_seed(seed)


class DataDistribution(object):   # 真实数据分布（蓝色的线）
    def __init__(self):
        self.mu = 4  # 均值
        self.sigma = 0.5  # 标准差

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):   # G网络的输入，随机噪声分布
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        # 均匀分布
        return np.linspace(-self.range, self.range, N) + \
               np.random.random(N) * 0.01  # 随机0-1
        '''
        samples = np.random.normal(4, 0.5, N)
        samples.sort()
        return samples
        '''


def linear(input, output_dim, scope=None, stddev=1.0):  # w和b参数的初始化#线性计算，计算y=wx+b
    norm = tf.random_normal_initializer(stddev=stddev)  # 用高斯的随机初始化给w进行初始化
    const = tf.constant_initializer(0.0)  # 用常量0给b进行初始化
    with tf.variable_scope(scope or 'linear'):  # 变量域为scope（默认继承外层变量域）的值,当值为None时，域为linear
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)  # input.get_shape()[1]获取input的列数
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def generator(input, h_dim):  # 生成网络
    # h0 = tf.nn.tanh(linear(input, h_dim, 'g0'))
    # h0 = tf.nn.sigmoid(linear(input, h_dim, 'g0'))
    # h0 = tf.nn.relu(linear(input, h_dim, 'g0'))  # 较好
    # h1 = tf.nn.relu(linear(h0, h_dim, 'g1'))#
    # h2 = linear(h1, 1, 'g2')
    # return h2
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))#原
    h1 = linear(h0, 1, 'g1')  # 原
    return h1  # 原


def discriminator(input, h_dim):  # 初始判别网络
    h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))  # 第一层的输出
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))
    h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))  # 使用sigmod激活函数将最终输出结果固定在0-1之间，方便对最终结果的真假概率进行计算
    return h3  #


def optimizer(loss, var_list, initial_learning_rate):  # 学习率不断衰减
    decay = 0.95
    num_decay_steps = 150  # 每迭代150次进行一次衰减，
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(  # 使用梯度下降求解器来最小化loss值，对var_list中的变量进行优化
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer


class GAN(object):  # 模型
    def __init__(self, data, gen, num_steps, batch_size, log_every):
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.mlp_hidden_size = 4  # 隐藏层神经元的个数

        self.learning_rate = 0.03  # 学习率

        self._create_model()

    def _create_model(self):

        with tf.variable_scope('D_pre'):  # 初始判别网络，变量域 D_pre
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size)  # 初始化参数，并获得网络的输出结果（0-1）之间的一个值
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))  # D_pre的输出结果和lalels的差异，或者说损失值
            self.pre_opt = optimizer(self.pre_loss, None,
                                     self.learning_rate)  # 使用衰减后的学习率，最小化loss，获得预先训练的网络模型，用其来初始化判别网络。

        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('Gen'):  # 生成网络
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))  # 噪声输入
            self.G = generator(self.z, self.mlp_hidden_size)  # 输出结果

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('Disc') as scope:  # 判别网络
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.mlp_hidden_size)  # 输入self.x为真实数据
            scope.reuse_variables()  # 变量重用
            self.D2 = discriminator(self.G, self.mlp_hidden_size)  # 输入self.G为生成数据

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.loss_d = tf.reduce_mean(
            -tf.log(self.D1) - tf.log(1 - self.D2))  # 判别网络损失函数#(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))  # 生成网络损失函数

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')  # 获取变量集合
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)  # 使用求解器对变量进行优化
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        with tf.Session() as session:
            tf.global_variables_initializer().run()  # 初始化变量

            # pretraining discriminator
            num_pretrain_steps = 1000  # 1000训练D_pre判别网络次数（预训练判别网络）
            for step in range(num_pretrain_steps):
                d = (np.random.random(
                    self.batch_size) - 0.5) * 10.0  # （判别网络真实数据输入值）随机生成初始数据，numpy.random.random(size = None): Return random floats in the half-open interval [0.0, 1.0),size一个参数时对应（列数），两个参数时对应（行，列）
                labels = norm.pdf(d, loc=self.data.mu,
                                  scale=self.data.sigma)  # （判别网络真实数据输出值）根据d进行高斯值的生成，norm.pdf(x, loc, scale) is identically equivalent to norm.pdf(y) / scale with y = (x - loc) / scale.
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt],
                                               {  # pretrain_loss, _ =用来接收返回值，如果返回值无用可以用 _ 简单表示
                                                   self.pre_input: np.reshape(d, (self.batch_size, 1)),
                                                   self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                                               })  # 完成D_pre的训练
            print(pretrain_loss)
            self.weightsD = session.run(self.d_pre_params)  # 获取训练后的参数
            # copy weights from pre-training over to new D network
            for i, v in enumerate(self.d_params):  # 把训练后的参数赋值给新的判别网络，enumerate:枚举
                session.run(v.assign(self.weightsD[i]))

            for step in range(self.num_steps):
                # update discriminator更新判别器
                x = self.data.sample(self.batch_size)  # 真实数据输入
                z = self.gen.sample(self.batch_size)  # 噪音输入
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # update generator更新生成器
                z = self.gen.sample(self.batch_size)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))  # 输出loss值
                if step % 100 == 0 or step == 0 or step == self.num_steps - 1:
                    self._plot_distributions(session)  # 绘制效果图

    def _samples(self, session, num_points=10000, num_bins=100):
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)  # 生成样本数量为10000的从-8到8的均匀数值数组
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # data distribution数据分布
        d = self.data.sample(num_points)  # 真实数据的分布，10000行，1列
        pd, _ = np.histogram(d, bins=bins, density=True)  # 根据输入数据获取一个样本数为100的具有相同数据分布的数组

        # generated samples生成的样本
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))  # 生成具有10000行1列的全为0的矩阵（初始化g）
        for i in range(num_points // self.batch_size):  # 取整除（返回商的整数部分）
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })  # 根据输入zs，获取生成网络生成的数据，10000行，1列
        pg, _ = np.histogram(g, bins=bins, density=True)

        return pd, pg

    def _plot_distributions(self, session):
        pd, pg = self._samples(session)  # 接收绘图所需数据数组，分别对应真实数据分布和生成数据分布
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))  # put pd to p_x
        f, ax = plt.subplots(1)  # 此功能通过一次调用创建一个图形和一个子图的网格，同时提供对各个图形创建方式的合理控制。参数为子图数量
        ax.set_ylim(0, 1)  # 设置y轴为0-1
        plt.plot(p_x, pd, label='real data')  # 绘制pd
        plt.plot(p_x, pg, label='generated data')  # 绘制pg
        plt.title('1D Generative Adversarial Network')  # 绘制标题
        plt.xlabel('Data values')  # x轴标题
        plt.ylabel('Probability density')  # y轴标题
        plt.legend()  # 添加图例（右上部位的线段加标签）
        plt.show()  # 显示图像


def main(args):
    model = GAN(
        DataDistribution(),  # 真实数据分布
        GeneratorDistribution(range=8),  # G网络的输入，噪声数据分布
        args.num_steps,  # 迭代次数
        args.batch_size,  # 一次迭代的数据点的个数
        args.log_every,  # 隔多少次打印loss值
    )
    model.train()


def parse_args():  # 参数列表
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=2000,  # 1200#生成网络训练次数
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,  # 每次（一批）需要的训练数据的数量
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,  # 输出loss值的间隔步数
                        help='print loss after this many steps')
    return parser.parse_args()


# python gan_normal.py
if __name__ == '__main__':
    main(parse_args())
