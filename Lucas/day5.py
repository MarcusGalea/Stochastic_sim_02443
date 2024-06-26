#%%
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
import math as math
import matplotlib.pyplot as plt
#%%

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
n = 100000
m = 40
areas = []
ps = []
#for _ in range(m):
samples = WHalgo(g, gparms, h, hparms, n, x0)
n_burn = 1000
n_actual = n - n_burn

samples = samples[n_burn:]
# Keep only every 25th sample
samples = samples[::50]
n_actual = len(samples)
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
#ps.append(1 - stats.chi2.cdf(T, df))
p = (1 - stats.chi2.cdf(T, df))
print(T, p)

# #%%
# print("Mean p-value is:", np.mean(ps))
# ps_sorted = np.sort(ps)
# print("Approximated CI for the p-value:")
# print(ps_sorted[2], ps_sorted[37])

# %%
plt.hist(samples, bins = 11)
plt.hist(expected, bins = 11)

#%% 2
def g(x,y,xparms):
    A1 = xparms[0]
    A2 = xparms[1]
    t1 = int(x)
    t2 = int(y)
    return (A1**t1 / math.factorial(t1)) * (A2 ** t2 / math.factorial(t2))
def h(hparms):
    U = rnd.uniform(size = 1)
    m = hparms[0]
    if U < 0.5:
        x = rnd.randint(low = 0, high = m+1)
        y = rnd.randint(low = 0, high = m+1 - x)
    else:
        y = rnd.randint(low = 0, high = m+1)
        x = rnd.randint(low = 0, high = m+1 - y)
    if x + y > 10:
        print("Error")
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
            if stats.uniform.rvs() <= 0.5:
                xs.append(x[0])
                ys.append(y[0])
            else:
                xs.append(y[0])
                ys.append(x[0])
    return np.array(xs, dtype = int), np.array(ys, dtype = int)

x0 = [0,0]
n = 500000
n_burn =1000
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

# Keep only every 50th sample
xs = xs[::75]
ys = ys[::75]

xs = np.array(xs ,dtype = int)
ys = np.array(ys, dtype = int)
xs_h = np.histogram(xs, bins = 11)[0]
ys_h = np.histogram(ys, bins = 11)[0]
print(ys_h)
print(xs_h)
#%%

print(stats.chisquare(f_obs = xs_h, f_exp = ys_h))
plt.hist(xs, bins = 11, alpha = 0.5, label='I-values')
plt.hist(ys, bins = 11, alpha = 0.5, label='J-values')
plt.legend(loc = 'upper right')
# #%% Check the sample

# n = len(xs)
# xs_true, ys_true = sampleDoublePoisson(n, A1, A2, m)
# print(np.histogram(xs_true, bins = 11)[0])
# print(np.histogram(ys_true, bins = 11)[0])
# plt.hist(xs_true, bins = 11, alpha = 0.5, label='I-values')
# plt.hist(ys_true, bins = 11, alpha = 0.5, label='J-values')
# plt.title('True Values')
# plt.legend(loc = 'upper right')

#%% Checking with Chi2 dist
n = len(xs)
def density(i,j):
    c = 0.00041121256417273044
    c = np.exp(-(A1 + A2))
    return c * A1**i / math.factorial(i) * A2**j / math.factorial(j)

dens = np.zeros((m+1, m+1))
for i in range(m+1):
    for j in range(m-i+1):
        dens[i,j] = density(i,j)
dens = dens / np.sum(dens)

plt.imshow(dens)
plt.colorbar()
#print(dens)

true = np.zeros((m+1, m+1), dtype = int)
true = dens * n

# ys_true = sum(dens)
# ys_true = ys_true * n
# ys_true[3] += 4
# ys_true = np.array(ys_true, dtype = int)
# print(sum(ys_true))
# print(sum(ys_h))
# print(stats.chisquare(f_obs = ys_h, f_exp = ys_true))

#%%
#test = np.zeros((m+1, m+1), dtype = int)
test = np.histogram2d(xs.astype(int), ys.astype(int), bins = 11)[0]
#true = np.histogram2d(xs_true, ys_true, bins = 11)[0]
plt.imshow(test)
plt.colorbar()
plt.show()

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

print("p-value is:", ps)
ps_sorted = ps
# print("Approximated CI for the p-value:")
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

print("Test Statisic is T =", T)



#%% c Gibbs sampling
true = np.array([[  20.,   84.,  149.,  189.,  236.,  164.,  117.,   67.,   26.,
          23.,    7.],
       [  74.,  320.,  677.,  842.,  887.,  710.,  471.,  268.,  143.,
          75.,    0.],
       [ 189.,  701., 1333., 1796., 1726., 1445.,  940.,  520.,  238.,
           0.,    0.],
       [ 242.,  889., 1718., 2373., 2269., 1844., 1210.,  702.,    0.,
           0.,    0.],
       [ 238.,  876., 1774., 2272., 2396., 1971., 1273.,    0.,    0.,
           0.,    0.],
       [ 172.,  646., 1404., 1793., 1959., 1473.,    0.,    0.,    0.,
           0.,    0.],
       [ 116.,  464.,  943., 1255., 1216.,    0.,    0.,    0.,    0.,
           0.,    0.],
       [  71.,  251.,  526.,  710.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.],
       [  29.,  129.,  276.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.],
       [  17.,   60.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.],
       [   6.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
           0.,    0.]])
def Gibbs(As, n, x0, m):
    xs = [x0[0]]
    ys = [x0[1]]
    A1 = As[0]
    A2 = As[1]
    for k in range(1,n):
        # Generate i and sample from j
        i = xs[k-1]
        num_classes_j = int(m - i + 1)
        ps = np.zeros(num_classes_j)
        k = 0
        for j in range(num_classes_j):
            ps[j] = A2**j / math.factorial(j)
            k += A2**j / math.factorial(j)
        ps /= k
        j = rnd.choice(a=np.arange(num_classes_j), size= 1, p = ps)[0]

        ys.append(j)
        # Newest j has already been found
        num_classes_i = int(m-j + 1)
        ps = np.zeros(num_classes_i )
        k = 0
        for i in range(num_classes_i):
            ps[i] = A2**i / math.factorial(i)
            k += A2**i / math.factorial(i)
        ps /= k
        i = rnd.choice(a=np.arange(num_classes_i), size=1, p = ps)[0]
        xs.append(i)
    return np.array(xs), np.array(ys)

def h(hparms):
    m = hparms[0]
    return np.randint(low = 0, high = m + 1, size = 0)
As = [4, 4]
x0 = [3,3]
n = 10000
m = 10
n_burn = 1000
xs, ys = Gibbs(As, n + n_burn, x0, m)
xs = xs[n_burn:]
ys = ys[n_burn:]

#%%
sample_hist = np.histogram2d(xs,ys, bins = 11)[0]
plt.imshow(sample_hist)

# Chisq test
sample_ints =sample_hist/ n
true_ints = true / n
dof = 66

sq = (sample_ints - true_ints)**2
T = sum(sum(np.divide(sq, true_ints, where = true_ints > 0)))

p = 1 - stats.chi2.cdf(T, df = dof)
print("Test statistic Gibbs sampling T =", T)
print("P-value using Gibbs sampling =", p)





# %% Exercise 3
# 1 
# Generate ksi and gammam standard normal distribution
def xigamma():
    cov = np.array([[1, 1/2], [1/2, 1]])
    mean = np.array([0, 0])
    x = rnd.multivariate_normal(mean, cov, 1)[0]
    return x[0], x[1]

def thetapsi():
    xi, gamma = xigamma()
    return np.exp(xi), np.exp(gamma)

theta, psi = thetapsi()
print([theta, psi])


# %%
# b 
# 
def getXs(theta, psi, n):
    return rnd.normal(loc = theta, scale = psi, size = n)
n = 10
Xs = getXs(theta, psi, n)
print(Xs)
# %% c

def f(Theta,Psi):
    rho = 0.5
    c = 1 / (2 * np.pi * Theta * Psi * np.sqrt(1 - rho**2))
    lx = np.log(Theta)
    ly = np.log(Psi)
    u = (lx **2 - 2 * rho * lx * ly + ly**2) / (2 * (1 - rho**2))
    return c * np.exp(-u)

def lh(x, Theta, Psi):
    ps = sum(np.log(stats.norm.pdf(x, loc = Theta, scale = Psi)))
    return np.exp(ps)

def h(hparms):
    mu = hparms[0]
    sd = hparms[1]
    Theta = stats.norm.rvs(loc = mu, scale = sd)
    while Theta < 0:
        Theta = stats.norm.rvs(loc = mu, scale = sd)
    Psi =  stats.norm.rvs(loc = mu, scale = sd)
    while Psi < 0:
        Psi = stats.norm.rvs(loc = mu, scale = sd)
    return Theta, Psi

def g(Theta, Psi, x):
    return f(Theta, Psi) * lh(x, Theta, Psi)


def ThetaPsiMH(g, gparms, h, hparms,  n, x0):
    Thetas = []
    Psis = []
    Thetas.append(x0[0])
    Psis.append(x0[1])
    Thetaim1 = Thetas[0]
    Psiim1 = Psis[0]
    p_old = g(Thetaim1,Psiim1, gparms)
    for _ in range(1,n):
        Thetai,Psii = h(hparms)
        p_new = g(Thetai, Psii, gparms)
        U = rnd.uniform(size = 1)
        if U <= min(1, p_new/p_old):
            Thetas.append(Thetai)
            Psis.append(Psii)
            Thetaim1 = Thetai
            Psiim1 = Psii
            p_old = p_new
        else:
            Thetas.append(Thetaim1)
            Psis.append(Psiim1)
    return np.array(Thetas), np.array(Psis)
#%%
x0 = [2,2]
hparms = [0, 0.5]
m = 5000
# set seed to 0 in numpy
#rnd.seed(1231232112)

theta, psi = thetapsi()
Thetas_plot = np.linspace(0.0005,14, 100)
Psis_plot = np.linspace(0.0005, 14, 100)
Zs = np.zeros((len(Thetas_plot), len(Psis_plot)))
for i in range(len(Thetas_plot)):
    for j in range(len(Psis_plot)):
        Zs[i,j] = f(Thetas_plot[i], Psis_plot[j])

for n in [10, 100, 1000]:
    gparms = getXs(theta, psi, n)
    print(np.mean(gparms))
    Thetas, Psis = ThetaPsiMH(g, gparms, h, hparms, m, x0)
    plt.hist(Thetas, density= True, bins = 20)
    plt.title(f'Theta for n = {n}')
    plt.show()
    plt.hist(Psis, density=True, bins = 20)
    plt.title(f'Psi for n = {n}')
    plt.show()
    plt.hist2d(Thetas, Psis)
    plt.contour(Thetas_plot, Psis_plot, Zs)
    plt.show()



# %%

A1 = 4
A2 = 4
m = 10
P = 0

for i in range(m):
    for j in range(m-i + 1):
        P += A1**i / math.factorial(i) * A2**j / math.factorial(j)
print(P)
c = 1 / P
print(c)




plt.imshow(dens)



# %%
