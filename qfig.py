import simpy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate

def client(env, lamda, q, tic):
    meant = 1/lamda
    while True:
        t = np.random.exponential(meant)
        yield env.timeout(t)
        q.put('job')
        tic.append(env.now)

def server(env, alpha, mu1, mu2, q, toc):
    mean1 = 1/mu1
    mean2 = 1/mu2
    while True:
        yield q.get()
        p = np.random.uniform()
        if p < alpha:
            t = np.random.exponential(mean1)
        else:
            t = np.random.exponential(mean2)
        yield env.timeout(t)
        toc.append(env.now)

lamda = 75
alpha = 0.333
mu1 = 370
mu2 = 370*(0.666)
num_bins = 50
runtime = 1000 #运行多长时间

tic = [] #每个任务进系统的时间点
toc = [] #每个任务出系统的时间点
env = sp.Environment()
q = sp.Store(env)

env.process(client(env, lamda, q, tic))
env.process(server(env, alpha, mu1, mu2, q, toc))
env.run(until=runtime)

l = len(tic)
a = toc
b = toc
#b = toc[0:l:40]

histdata = [b[i] - b[i-1] for i in range(1, len(b))]
sns.distplot(histdata, kde=False, fit=stats.expon)
plt.xlabel("inter departure time (s)")
plt.xlim(0,0.15)
#plt.ylim(0,100)
plt.savefig('dist1.png')
plt.show()

#plt.hist(histdata, num_bins)
#plt.show()


