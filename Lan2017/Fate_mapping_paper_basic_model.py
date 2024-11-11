import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math

init_population=np.array([1,0]) #inital condition
population=init_population
#the effect each transition has on the states:
state_changes= np.array([
        [0,1], #The stem cell divides asymmetrically
        [0,1], #The progenitor cells divides symmetrically
        [0,-1], #The progenitor cell dies
    ])
l=[]
logl=[]
time_points=[]
def gillespie(t_start,t_stop,omega, lam, population, state_changes,l,logl,time_points):
    t=t_start
    while t<t_stop:
        s,p=population 

        prob=np.array([omega*s, 0.5*lam*p, 0.5*lam*p]) #probability of each transition happening

        del_t=np.random.exponential(1/(np.sum(prob))) #choosing a time change from an exponential distribution

        state=scipy.stats.rv_discrete(values=([0,1,2],[prob[0]/np.sum(prob), prob[1]/np.sum(prob), prob[2]/np.sum(prob)])).rvs() #choosing a state change from a discrete distribution 
        t=t+del_t #updating the time
        population=population+state_changes[state] #updating the population
        l.append(p+1)
        logl.append(math.log(p+1))
        time_points.append(t)
    return population, l,logl, time_points

passage_times=np.array([0,500,1000,1500])
passage_start_amounts=np.array([[1,0],[1,2],[1,2]])

for j in range(5):
    for i in range(passage_times.size-1):
        population,l,logl,time_points=gillespie(t_start=passage_times[i],t_stop=passage_times[i+1],omega=0.07,lam=0.7,population=passage_start_amounts[i], state_changes=state_changes, l=l,logl=logl, time_points=time_points)
        if time_points[-1]>passage_times[i+1]:
            del l[-1]
            del logl[-1]
            population=[1,l[-1]]
            del time_points[-1]
        plt.plot(time_points,logl)
    l=[]
    logl=[]
    time_points=[]

plt.ylabel('Clone Size')
plt.xlabel('Time')
plt.axvline(500)
plt.axvline(1000)
plt.title('Model simulations on a logarithmic scale')
plt.show()
plt.clf()