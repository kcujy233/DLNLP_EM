import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
import pandas as pd

class GMM:
    def __init__(self, Data, K, data_m, weights = None,means = None,covars = None):
        """
        构造GMM（高斯混合模型）的类
        Data: 训练数据
        K: 高斯分布的个数
        weigths: 每个高斯分布的初始概率（权重）
        means: 高斯分布的均值向量
        covars: 高斯分布的协方差矩阵集合
        """
        self.Data = Data                                #定义数据集
        self.K = K                                      #定义高斯分布个数
        col = np.shape(self.Data)[1]                    #输入数据的特征维度
        self.data_m = data_m
        if weights is not None:                         #权重赋值
            self.weights = weights
        else:
            self.weights  = np.random.rand(self.K)
            self.weights /= np.sum(self.weights)        # K个维度的归一化

        if means is not None:                           #均值为K个维度
            self.means = means
        else:
            self.means = []
            for i in range(self.K):                     #对K个维度分别求平均值
                mean = np.random.rand(col)
                mean = mean / np.sum(mean)              # 归一化
                self.means.append(mean)

        if covars is not None:
            self.covars = covars
        else:
            self.covars  = []
            for i in range(self.K):
                cov = np.random.rand(col,col)
                cov = cov / np.sum(cov)                 # 归一化
                self.covars.append(cov)                 # cov是np.array,但是self.covars是list

    def Gaussian(self, x, mean, cov):
        """
        自定义的高斯分布概率密度函数
        x: 输入数据
        mean: 均值数组
        cov: 协方差矩阵
        :return:返回概率密度
        """
        dim = np.shape(cov)[0]
        # 为了防止cov的行列式为零，让该行列式加上一个小的值
        # covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        # covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        #为了保证测试结果，选择舍弃增加偏置的方案
        covdet = np.linalg.det(cov)
        covinv = np.linalg.inv(cov)
        xdiff = (x - mean).reshape((1, dim))
        # 概率密度函数
        dense = 1.0/(np.power(np.power(2*np.pi, dim)*np.abs(covdet), 0.5)) * \
                np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return dense

    def GMM_EM(self):
        """
        利用EM算法解算GMM参数的函数
        :return:返回各组数据属于每个分类的概率
        """
        loglikelyhood = 0
        oldloglikelyhood = 1
        len, dim = np.shape(self.Data)                      #数据的长宽
        # gamma表示第n个样本属于第k个混合高斯的概率
        gammas = [np.zeros(self.K) for i in range(len)]
        while np.abs(loglikelyhood-oldloglikelyhood) > 0.001:
            oldloglikelyhood = loglikelyhood
            # E-step
            for n in range(len):
                # respons是GMM的EM算法中的权重w，即后验概率
                respons = [self.weights[k] * self.Gaussian(self.Data[n], self.means[k], self.covars[k])
                                                    for k in range(self.K)]
                respons = np.array(respons)
                sum_respons = np.sum(respons)
                gammas[n] = respons/sum_respons
            # M-step
            for k in range(self.K):
                #nk表示N个样本中有多少属于第k个高斯
                nk = np.sum([gammas[n][k] for n in range(len)])
                # 更新每个高斯分布的概率
                self.weights[k] = 1.0 * nk / len
                # 更新高斯分布的均值
                self.means[k] = (1.0/nk) * np.sum([gammas[n][k] * self.Data[n] for n in range(len)], axis=0)
                xdiffs = self.Data - self.means[k]
                # 更新高斯分布的协方差矩阵
                self.covars[k] = (1.0/nk)*np.sum([gammas[n][k]*xdiffs[n].reshape((dim,1)).dot(xdiffs[n].reshape((1,dim))) for n in range(len)],axis=0)
            loglikelyhood = []
            for n in range(len):
                tmp = [np.sum(self.weights[k]*self.Gaussian(self.Data[n],self.means[k],self.covars[k])) for k in range(self.K)]
                tmp = np.log(np.array(tmp) + 1e-5)
                loglikelyhood.append(list(tmp))
            loglikelyhood = np.sum(loglikelyhood)
        for i in range(len):
            gammas[i] = gammas[i]/np.sum(gammas[i])
        self.posibility = gammas
        self.prediction = [np.argmax(gammas[i]) for i in range(len)]

def get_data():
    '''
    获取数据
    划分为数据以及用以标准检测的标签
    标签转化为int32的形式
    :return:返回数据和标签以及最大数值
    '''
    oridata = pd.read_csv('height_data.csv', usecols=[0])
    data = np.array(oridata).astype(float)
    data_m = np.max(data).astype(float)
    data /= data_m                                          #对数据进行归一化
    return data, data_m

def run_main():
    """
    主函数
    """
    # 导入数据集
    data, data_m = get_data()
    x = list(range(1, len(data)+1))
    x = np.array(x).reshape(2000,1).astype(float)

    # 解决画图是的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    #
    # # 数据可视化
    plt.scatter(x, data)
    plt.title("身高数据集显示")
    plt.show()
    x /= data_m
    dt = np.concatenate((x, data), axis=1)

    # # GMM模型
    K = 2
    gmm = GMM(dt, K, data_m)
    gmm.GMM_EM()
    y_pre = gmm.prediction

    print("预测结果：\n", y_pre)
    plt.scatter(dt[:, 0], dt[:, 1], c=y_pre)
    plt.title("分布预测结果显示")
    plt.show()

if __name__ == '__main__':
    run_main()