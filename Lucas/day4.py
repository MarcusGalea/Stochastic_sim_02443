#%%
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
import math as math

#%% 1 Crude Motecarlo
def crudeMC(n):
    U = rnd.uniform(size = n)
    x = np.exp(U)
    return x

def meanVar(x):
    mean = np.mean(x)
    var = np.var(x)
    return mean, var

def normConf(x, alpha, string):
    mean, var = meanVar(x)
    s = np.sqrt(var / len(x))
    a = stats.norm.ppf(alpha/2)
    b = stats.norm.ppf(1 - alpha/2)
    print("For", string, ":")
    print(f"Mean is {round(mean,2)}, with confidence interval [{round(mean + s * a,2)},{round(mean + s * b,2)}]")

alpha = 0.05
n = 100
Xs_crude = crudeMC(n)
normConf(Xs_crude, alpha, "Crude Monte Carlo" )
# %% Antithetic variables
def antithetic(n):
    U = rnd.uniform(size = n)
    t = np.exp(U)
    Ys = 0.5 * (t + np.exp(1) / t )
    return Ys
Xs_antithetic = antithetic(100)
normConf(Xs_antithetic, alpha, "Antithetics Variables")

# %% Control Variates
def controlVariates(n, alpha):
    U = rnd.uniform(size = n)
    Xs = np.exp(U)
    meanZ = np.mean(Xs)
    cov = np.mean(U * Xs) - np.mean(U) * np.mean(Xs) 
    varZ = np.var(Xs) - cov**2 / np.var(U)
    s = np.sqrt(varZ / len(Xs))
    a = stats.norm.ppf(alpha/2)
    b = stats.norm.ppf(1 - alpha/2)
    print("For Control Variates")
    print(f"Mean is {round(meanZ,2)}, with confidence interval [{round(meanZ + s * a,2)},{round(meanZ + s * b,2)}]")

controlVariates(n, alpha)

#%% Stratified Sampling
def stratSamples(n):
    num_ints = 10
    k = 1 / num_ints
    m = 0
    Xs = np.zeros(n)
    for i in range(num_ints):
        U = rnd.uniform(size = int(n / num_ints))
        a = k * i
        b = a + k 
        U = U * (b - a) + a
        Xs[i * 10: (i+1)*10] = np.exp(U)
    return Xs
Xs_stratified = stratSamples(n)
normConf(Xs_stratified, alpha, "Stratified Sampling")


# %% 5 Changing Marcuss code
from scipy.stats import poisson
#import exponential
from scipy.stats import expon
import bisect

m = 10 #number of servers
s = 8 #mean service time
lam = 1#arrival_intensity
total_customers =10000 #10*10000
A = lam*s

class Customer:
    def __init__(self, arrival_time, service_time):
        self.service_time = service_time
        self.blocked = False
        
        self.event = "arrival"
        self.event_time = arrival_time
      
                
    def arrive(self, servers, event_list):
        if servers < 1:
            self.blocked = True
            return servers
        else:
            servers -= 1
            servers = max(servers, 0)
            self.event = "departure"
            self.event_time += self.service_time
            bisect.insort(event_list, self, key=lambda x: x.event_time)
            return servers
    
    def depart(self, servers):
        servers += 1
        servers = min(servers, m)
        return servers

from distributions import getExponential, getUniform

def transformToZ(parms):
    lam = parms[0]
    n = parms[1]
    Us = rnd.uniform(0, 1, n)
    Xs = -np.log(Us) / lam
    cov = np.cov(Us, Xs)[0,1]
    c = - cov / np.var(Us)
    Zs = Xs + c * (Us - 1/2)
    return Zs


def main_loop(event_list, m, repititions = 10):

    blocked = 0
    for i in range(repititions):
        event_list.sort(key=lambda x: x.event_time)
        open_servers = m
        while event_list:
            event = event_list.pop(0)
            if event.event == "arrival":
                open_servers = event.arrive(open_servers, event_list)
                blocked += event.blocked
            elif event.event == "departure":
                open_servers = event.depart(open_servers)
    return blocked

reps = 40
blocks = np.zeros(reps)
for i in range(reps):
    arrival_intervals = transformToZ([lam, total_customers])
    arrival_times = np.cumsum(arrival_intervals)
    service_intervals = transformToZ([1/s, total_customers])
    event_list = [Customer(arrival_times[i], service_intervals[i]) for i in range(total_customers)]
    blocks[i] = main_loop(event_list, m, repititions= 10)

# %%
blocked_saved = np.array([504, 547, 517, 579, 520, 536, 527, 567, 531, 561, 527,472 ,501, 556,489 ,569, 570, 495, 511, 530, 491, 560, 543, 499, 481, 558 ,565, 569,500, 580, 497, 462, 505, 549, 538, 553, 552, 489, 521, 562])

print(f"Mean Percent Blocked {np.round(np.mean(blocked_saved) / total_customers * 100,3)}%")
mean = np.mean(blocked_saved) / total_customers * 100
var = np.var(blocked_saved / total_customers * 100)
s = np.sqrt(var / len(blocked_saved))
a = stats.norm.ppf(alpha/2)
b = stats.norm.ppf(1 - alpha/2)
print(f"Mean is {round(mean,3)}%, with confidence interval [{round(mean + s * a,3)},{round(mean + s * b,3)}]%")

# %% Common random numbers
# We're going to solve the model with two types of arrival, either poisson for system 1 or hyperexponential for system 2

num_customers = 10000
lam = 1
s = 1/8
# Generate two eventlists
p1 = 0.8
lam1 = 0.8333
p2 = 0.2
lam2 = 5.0

def generateHexp(Us, Us2, p2, lam1, lam2):
    Xs = np.zeros(len(Us))
    for i in range(len(Us)):
        if Us2[i] < p2:
            Xs[i] = -np.log(Us[i]) / lam2
        else:
            Xs[i] = -np.log(Us[i]) / lam1
    return Xs

m = 10
rep = 5
blocked_hexp = np.zeros(rep)
blocked_poisson = np.zeros(rep)
blocked_poisson_common = np.zeros(rep)
blocked_hexp_common = np.zeros(rep)
for i in range(rep):

    Us1 = rnd.uniform(0,1,num_customers)
    Us2 = rnd.uniform(0,1,num_customers)
    Us3 = rnd.uniform(0,1,num_customers)
    service_intervals = -np.log(Us1) / s
    poisson_arrival_times = np.cumsum(-np.log(Us2) / lam)
    hexp_arrival_times = np.cumsum(generateHexp(Us2, Us3, p2, lam1, lam2))

    event_list_poisson = [Customer(poisson_arrival_times[i], service_intervals[i]) for i in range(num_customers)]
    event_list_hexp = [Customer(hexp_arrival_times[i], service_intervals[i]) for i in range(num_customers)]

    blocked_poisson_common[i] = main_loop(event_list_poisson, m, repititions=1)
    blocked_hexp_common[i] = main_loop(event_list_hexp, m, repititions=1)

    Us1 = rnd.uniform(0,1,num_customers)
    Us2 = rnd.uniform(0,1,num_customers)
    Us3 = rnd.uniform(0,1,num_customers)
    Us4 = rnd.uniform(0,1,num_customers)
    Us5 = rnd.uniform(0,1,num_customers)
    service_intervals1 = -np.log(Us2) / s
    service_intervals2 = -np.log(Us3) / s

    poisson_arrival_times = np.cumsum(-np.log(Us1) / lam)
    hexp_arrival_times = np.cumsum(generateHexp(Us4, Us5, p2, lam1, lam2))

    event_list_poisson = [Customer(poisson_arrival_times[i], service_intervals1[i]) for i in range(num_customers)]
    event_list_hexp = [Customer(hexp_arrival_times[i], service_intervals2[i]) for i in range(num_customers)]

    blocked_poisson[i] = main_loop(event_list_poisson, m, repititions=1)
    blocked_hexp[i] = main_loop(event_list_hexp,m, repititions=1)

#%%
theta_crn = blocked_poisson_common - blocked_hexp_common
theta_irn = blocked_poisson - blocked_hexp

mean_theta_crn = round(np.mean(theta_crn),0)
var_theta_crn = np.var(theta_crn)
s = np.sqrt(var_theta_crn / len(theta_crn))
a = stats.norm.ppf(alpha/2)
b = stats.norm.ppf(1 - alpha/2)
print("For common random numbers")
print(f"Mean is {mean_theta_crn}, with confidence interval [{round(mean_theta_crn + s * a,3)},{round(mean_theta_crn + s * b,3)}]")
print(f"Width of CI = {round(np.abs(mean_theta_crn + s * a - mean_theta_crn - s*b),3)}")


mean_theta_irn = int(np.mean(theta_irn))
var_theta_irn = np.var(theta_irn)
s = np.sqrt(var_theta_irn / len(theta_irn))
a = stats.norm.ppf(alpha/2)
b = stats.norm.ppf(1 - alpha/2)
print("For independent random numbers")
print(f"Mean is {mean_theta_irn}, with confidence interval [{round(mean_theta_irn + s * a,3)},{round(mean_theta_irn + s * b,3)}]")
print(f"Width of CI = {round(abs(mean_theta_irn + s * a - mean_theta_irn - s*b),3)}")

#%% 7
def crudeMonteCarloNorm(a,n):
    




    

 