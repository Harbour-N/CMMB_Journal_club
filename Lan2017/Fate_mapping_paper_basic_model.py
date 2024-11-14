import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math

init_population=np.array([1,0]) #inital condition (start with a single GSC (s) cell)
population=init_population
#the effect each transition has on the states:
state_changes= np.array([
        [0,1], #The stem cell divides asymmetrically
        [0,1], #The progenitor cells divides symmetrically
        [0,-1], #The progenitor cell dies
    ])

def gillespie(t_start,t_stop,omega,lam,state_changes,init_population,time):
    time_points=np.zeros(time)
    t=t_start
    sizes=np.zeros((time,2))
    sizes[0,:]=init_population

    i=0
    while time_points[i]<t_stop and i<time-1:
        time_points[0]=t
        s,p=sizes[i,:] 

        prob=np.array([omega*s, 0.5*lam*p, 0.5*lam*p]) #probability of each transition happening

        del_t=np.random.exponential(1/(np.sum(prob))) #choosing a time change from an exponential distribution

        state=scipy.stats.rv_discrete(values=([0,1,2],[prob[0]/np.sum(prob), prob[1]/np.sum(prob), prob[2]/np.sum(prob)])).rvs() #choosing a state change from a discrete distribution 
        
        time_points[i+1]=time_points[i]+del_t #updating the time
        sizes[i+1,:]=sizes[i,:]+state_changes[state] #updating the population
        
        i=i+1

    #removing any zero entries from the back
    time_points=np.trim_zeros(time_points, 'b')

    if time_points[-1]>t_stop:
        time_points=np.delete(time_points,-1)

    sizes=sizes[:time_points.size]
    return sizes, time_points

#simulations
for i in range(10):
    size, time_p=gillespie(0, 1000, 0.07, 0.1,state_changes=state_changes, init_population=init_population, time=1000)
    plt.semilogy(time_p,size[:,1])

plt.ylabel('Clone Size')
plt.xlabel('Time')
plt.title('Simulations of asymmetric division only model')
plt.show()
plt.clf()

#simulations with reinjections
passage_times=np.array([0,500,1000,1500])
passage_start_amounts=np.array([[1,0],[1,2],[1,2]])

for i in range(10):
    passage=np.array([])
    pass_times=np.array([])
    for i in range(passage_times.size-1):
        size, time_p=gillespie(passage_times[i], passage_times[i+1], 0.07,0.1,state_changes=state_changes,init_population=passage_start_amounts[i],time=passage_times[1]-passage_times[0])
        passage=np.append(passage,size[:,1])
        pass_times=np.append(pass_times,time_p)

    plt.semilogy(pass_times, passage)
    plt.axvline(passage_times[1])
    plt.axvline(passage_times[2])

plt.xlabel('Time')
plt.ylabel('Clone Size on a logarithmic scale')
plt.title('Simulations of the asymmetric only model with 3 passages')
plt.show()