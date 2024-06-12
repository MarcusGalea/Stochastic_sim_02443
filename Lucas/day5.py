#%%
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
import math as math
import matplotlib.pyplot as plt
#%%
def RWMH(f, g, n, X0):
    xs = np.zeros(n-1)
    xs[0] = X0
    for i in range(1,n):
        xi = xs[i-1]
        dx = f()
        yi = xi + dx
        gx = g(xi)
        gy = g(yi)
        if gy >= gx:
            xs[i] = yi
        else:
            U = rnd.uniform(size = 1)
            if U <= gy/gx:
                 xs[i] = yi
            else:
                xs[i] = xi
    return xs

A = 1

def g(x,y, gparms):
    A = gparms[0]
    t = int(x)
    return A**t/ math.factorial(t)

def h(hparms):
    m = hparms[0]
    return _, rnd.randint(low = 0, high = m+1)

def WHalgo(g, gparms, h, hparms,  n, x0):
    xs = np.zeros(n)
    xs[0] = x0
    for j in range(1,n):
        x = xs[j - 1]
        _,y = h(hparms)
        py = g(y,x, gparms)
        px = g(x,y, gparms)
        U = rnd.uniform(size = 1)
        if U <= min(1, py/px):
            xs[j] = y
        else:
            xs[j] = x 
    return xs
#%%
x0 = 3
hparms = [10]
gparms = [8]
n = 10000
m = 40
areas = []
ps = []
for _ in range(m):
    samples = WHalgo(g, gparms, h, hparms, n, x0)
    n_burn = 1000
    n_actual = n - n_burn

    samples = samples[n_burn:]
    expected = []
    while len(expected) < n_actual:
        new = stats.poisson.rvs(8, size = 1)
        if new <= 10:
            expected.append(new[0])

    expected = np.array(expected)
    hist_exp = np.histogram(expected)[0]
    hist_samples = np.histogram(samples)[0]
    T = sum((hist_samples - hist_exp)**2 / hist_exp)
    df = len(hist_samples) - 1
    ps.append(1 - stats.chi2.cdf(T, df))

#%%
print("Mean p-value is:", np.mean(ps))
ps_sorted = np.sort(ps)
print("Approximated CI for the p-value:")
print(ps_sorted[2], ps_sorted[37])

# %%
plt.hist(samples, bins = 11)
plt.hist(expected, bins = 11)

#%% 2
def g(x,y,xparms):
    A1 = xparms[0]
    A2 = xparms[1]
    t1 = int(x)
    t2 = int(y)
    return A1**t1 / math.factorial(t1) * A2 ** t2 / math.factorial(t2)
def h(hparms):
    U = rnd.uniform(size = 1)
    m = hparms[0]
    if U <= 0.5:
        x = rnd.randint(low = 0, high = m+1)
        y = rnd.randint(low = 0, high = m+1 - x)
    else:
        y = rnd.randint(low = 0, high = m+1)
        x = rnd.randint(low = 0, high = m+1 - y)
    return x,y

def MVWHalgo(g, gparms, h, hparms,  n, x0):
    vals = []
    vals.append(x0)
    for j in range(1,n):
        cs = vals[j - 1]
        xim1 = cs[0]
        yim1 = cs[1]
        x,y = h(hparms)
        p_new = g(x,y, gparms)
        p_old = g(xim1,yim1, gparms)
        U = rnd.uniform(size = 1)
        if U <= min(1, p_new/p_old):
            vals.append( [x,y])
        else:
            vals.append( [xim1,yim1])
    return np.array(vals)


def sampleDoublePoisson(n, A1, A2, m):
    xs = []
    ys = []
    while len(xs) < n:
        x = stats.poisson.rvs(A1, size = 1)
        y = stats.poisson.rvs(A2, size = 1)
        if y + x <= m:
            xs.append(x[0])
            ys.append(y[0])
    return xs, ys

x0 = [3,3]
n = 50000
n_burn =10000
A1 = 4
A2 = 4
m = 10
hparms = [m]
gparms = [A1, A2]

samples = MVWHalgo(g, gparms, h, hparms, n+n_burn, x0)
samples = samples[n_burn:]
xs = []
ys = []
for i in range(n):
    xs.append(samples[i][0])
    ys.append(samples[i][1])

xs = np.array(xs)
ys = np.array(ys)
plt.hist(xs, bins = 11, alpha = 0.5, label='I-values')
plt.hist(ys, bins = 11, alpha = 0.5, label='J-values')
plt.legend(loc = 'upper right')
#%% Check the sample
xs_true, ys_true = sampleDoublePoisson(n, A1, A2, m)
plt.hist(xs_true, bins = 11, alpha = 0.5, label='I-values')
plt.hist(ys_true, bins = 11, alpha = 0.5, label='J-values')
plt.title('True Values')
plt.legend(loc = 'upper right')

#%% Checking with Chi2 dist
xs_true = np.array(xs_true)
ys_true = np.array(ys_true)
test = np.histogram2d(xs, ys, bins = 11)[0]
true = np.histogram2d(xs_true, ys_true, bins = 11)[0]
plt.imshow(test)
plt.colorbar()
plt.show(
)

plt.imshow(true)
plt.colorbar()
T = 0
for i in range(m+1):
    for j in range(m+1):
        if true[i][j] == 0:
            continue
        else:
            T += (test[i][j] - true[i][j])**2 / true[i][j]

df = sum(sum(test != 0))
ps = 1 - stats.chi2.cdf(T, df)
print("Mean p-value is:", ps)
ps_sorted = ps
print("Approximated CI for the p-value:")
print("T-value", T)


#%%
def MVCWWH(g, gparms, h, hparms,  n, x0):
    xs = [x0[0]]
    ys = [x0[1]]
    for j in range(1,n):
        xim1 = xs[j-1]
        yim1 = ys[j-1]
        x,y = h(hparms)

        # X-direction
        p_new = g(x,yim1, gparms)
        p_old = g(xim1, yim1, gparms)

        U = rnd.uniform(size = 1)
        if U <= min(1, p_new/p_old):
            x_to_append = x
        else:
            x_to_append = xim1

        p_new = g(x_to_append,y, gparms)
        p_old = g(x_to_append,yim1, gparms)
        U = rnd.uniform(size = 1)
        if U <= min(1, p_new/p_old):
            y_to_append = y
        else:
            y_to_append = yim1
        xs.append(x_to_append)
        ys.append(y_to_append)
    return np.array(xs), np.array(ys)

x0 = [3,3]
A1 = 4
A2 = 4
m = 10
hparms = [m]
gparms = [A1, A2]

xs, ys = MVCWWH(g, gparms, h, hparms, n+n_burn, x0)
xs = xs[n_burn:]
ys = ys[n_burn:]

plt.hist(xs, bins = 11, alpha = 0.5, label='I-values')
plt.hist(ys, bins = 11, alpha = 0.5, label='J-values')
plt.legend(loc = 'upper right')
plt.title('Values from Coordinate Wise')

test = np.histogram2d(xs, ys, bins = 11)[0]
plt.show()
plt.imshow(test)
plt.colorbar()
T = 0
for i in range(m+1):
    for j in range(m+1):
        if true[i][j] == 0:
            continue
        else:
            T += (test[i][j] - true[i][j])**2 / true[i][j]

print("Test Statisic is", T)
# %%
