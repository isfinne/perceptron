# %%
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import random

gloss = []

np.random.seed(12)
num_observations = 500

x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]],
                                   num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]],
                                   num_observations)

x1_test = x1[0:100]
x1 = x1[100:]
x2_test = x2[0:100]
x2 = x2[100:]
X = np.vstack((x1,x2)).astype(np.float32)
Y = np.hstack((np.zeros(400),np.ones(400)))
X_test = np.vstack((x1_test,x2_test)).astype(np.float32)
Y_test = np.hstack((np.zeros(100),np.ones(100)))

# %%
def f(x):
    return 1 if x > 0 else 0

def imageToGif(inputName, outfileName):
    files = os.listdir(inputName)
    print(files)
    frames = []
    for file in files:
        frames.append(imageio.imread(inputName + '\\' + file))
    imageio.mimsave(outfileName, frames, 'GIF', duration=0.01)

# %%
class Perceptron(object):
    def __init__(self, X, Y, lr):  # 类中参数是 X,Y（X,Y)均为numpy数组，lr是学习率
        if X.shape[0] != Y.shape[0]:  # 要求X,Y中的数目一样，即一个x对应一个y，否则返回错误
            raise ValueError('Error,X and Y must be same when axis=0 ')
        else:  # 在类中储存参数
            # 将X中的每一行都添加一个1，即添加偏置项
            self.X = np.c_[np.ones(X.shape[0]), X]
            self.Y = Y
            self.lr = lr

    def fit(self):
        weight = np.zeros(self.X.shape[1])  # 初始化weight
        number = 0  # 记录训练次数
        Vis = Plotting(X, Y)
        while True:
            loss = 0 # 记录损失
            if number == 500:
                Vis.close()
                break
            index = [i for i in range(self.X.shape[0])] 
            random.shuffle(index)
            for j in range(self.X.shape[0]):
                yi = f(np.dot(self.X[index[j]], weight))
                weight += self.lr * (self.Y[index[j]] - yi) * self.X[index[j]]
                loss += np.abs(self.Y[index[j]] - yi)
            number += 1
            loss /= self.X.shape[0]
            Vis.open_in()
            Vis.vis_plot(weight[1:], weight[0],number,loss)
            gloss.append(loss)
            if loss < 0.001:
                Vis.close()
                break
        return weight  # 返回值是weight

    def measure(self,X_test,Y_test,w):
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        fault = 0
        FP = 0
        FN = 0
        for i in range(X_test.shape[0]):
            yi = f(np.dot(X_test[i],w))
            if yi != Y_test[i]:
                fault += 1
                if yi ==1:
                    FP += 1
                else:
                    FN += 1
        print('fault num: ',fault)
        
        fault = fault/X_test.shape[0]
        P = 1 - FP/100 
        print('P: ',P)
        R = 1 - FN/100
        print('R: ',R)
        print('F1: ',2*P*R/(P+R))


# %%
class Plotting(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def open_in(self):
        plt.ion()

    def close(self):
        plt.ioff()
        plt.show()

    def vis_plot(self, weight, b, number, loss):
        plt.cla()
        plt.xlim(-4, np.max(self.X.T[0]) + 1)
        plt.ylim(-4, np.max(self.X.T[1]) + 1)
        plt.scatter(self.X.T[0], self.X.T[1], c=self.Y)
        if True in list(weight == 0):
            plt.plot(0, 0)
        else:
            x1 = -b / weight[0]
            x2 = -b / weight[1]
            plt.axline([x1, 0], [0, x2])
        plt.title('change time:{}   loss:{:.4f}'.format(number,loss))
        number1 = "%05d"%number
        plt.savefig(r'pil\%s.png' % number1)
        plt.pause(0.01)

# %%
model = Perceptron(X, Y, 0.001)
w = model.fit()
model.measure(X_test,Y_test,w)
print(w)

plt.plot(list(range(len(gloss))),gloss)
plt.ylabel('loss')
plt.show()

imageToGif('pil', 'perceptron.GIF')


