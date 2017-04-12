import numpy as np
import matplotlib.pyplot as plt
W1=1
b1=1

W2=1
b2=1

def truncNorm(mu,sig,a=0,N=1000):
    ret = []
    for i in range(N):
        x=-1
        while x<a:
            x = np.random.randn(1)*sig+mu
        ret.append(x)

    ret = np.array(ret)
    return ret

X = truncNorm(-4,4.0,0,10000)
Z = truncNorm(0.0,1.0,0,10000)
Y = truncNorm(-2,2.0,0, 10000)

plt.figure()
plt.hist(X, bins=50)
plt.show()

plt.figure()
plt.hist(Z, bins=50)
plt.show()

plt.figure()
plt.hist(Y, bins=50)
plt.show()

plt.figure()
plt.hist(Y+Z, bins=50)
plt.show()

plt.figure()
plt.hist(X+Y+Z, bins=50)
plt.show()
