import math
from numpy.random import gumbel
import numpy as np 

def rbp(rel,gamma):
    rbp_score=0
    for i in range(len(rel)):
        rbp_score += rel[i] * math.pow( 1- gamma, i) 
    return ( gamma )  *  rbp_score

def err(rel,gamma):
    err_score=0
    for i in range(len(rel)):
        prod=1
        for j in range(i):
            prod *= 1 - rel[j]
        err_score += (rel[i] / (i+1) ) * math.pow(gamma , i )  * prod
    return err_score


def rbp_trajectory(rel,trajectory,gamma):
    rbp_score=0
    for i in range(len(rel)):
        rbp_score += rel[trajectory[i]] * math.pow( 1- gamma, i) 
    return ( gamma )  *  rbp_score

def err_trajectory(rel,trajectory,gamma):
    print('-0------')
    print(rel,trajectory,gamma)
    err_score=0
    for i in range(len(rel)):
        prod=1
        for j in range(i):
            prod *= 1 - rel[trajectory[j]]
        err_score += (rel[trajectory[i]] / (i+1) ) * math.pow(gamma , i )  * prod
        print(err_score)
    return err_score

def gumbel_max_sample(x):
    z = gumbel(loc=0, scale=1, size=x.shape)
    return (x + z).argmax()

def gumbel_max_sample_array(x):
    idices = list(range(len(x)))
    x_popped = x 
    sampled_path=[]
    for i in range(len(x)):
        z = gumbel(loc=0, scale=1, size=x_popped.shape)
        selected = (x_popped + z).argmax()
        sampled_path.append( idices[selected])
        x_popped=np.delete(x_popped,selected)
        idices.remove(idices[selected])
    return(sampled_path)

