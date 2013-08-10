################################################################################
import numpy
from soft_classification.utils.distances  import distance
from numpy import dot, array, sum, zeros, outer, any, apply_along_axis,argmax,ones,newaxis,zeros_like,transpose
from numpy.linalg import det,inv
#from math import exp 
from FCM import FuzzyCMeans
import math


################################################################################
# GG -FuzzyCmeans class
################################################################################
class GGFuzzyCMeans(FuzzyCMeans):

    def membership(self):

        from spectral import status

        (M, B) = self.x.shape
        C, _ = self.c.shape
        r = zeros((M, C))
        m1 = 1./(self.m-1.)
        if self.verbose:
            print 'iteration number %d ' % self.iter
            status.display_percentage('updating membership...')
            
        sumU = sum(self.mu,axis=0)   
        alfa = sumU / M
        fi = zeros((C,B,B))
        det_fi = ones((C))
        inv_fi = ones((C,B,B))
        for i in range(C):
            fi[i] = covariance_matrix(self.x,self.mu,self.c[i],self.m,i)
            det_fi[i] = float(det(fi[i]))** 0.5
            inv_fi[i] = inv((fi[i]))
        den = zeros((C))
        for k in range(M):
            for j in range(C):
                den[j] = dis_GG(self.x[k],self.c[j],fi[j],alfa[j],det_fi[j],inv_fi[j])
                if den[j] > 1.e30:
                    den[j] =1.e30
            if any(den == 0):
                return self.mu
            
            frac = outer(den, 1./den)**m1
            r[k, :] = 1. / sum(frac, axis=1)
            if self.verbose:
                status.update_percentage(float(k) / M * 100.)
        self.mu = r
        if self.verbose:
            status.end_percentage()
        return self.mu


def covariance_matrix(training_set, initial_conditions,centers,m,clas):
    
    (N, B) = training_set.shape
    u = initial_conditions ** m
    sumX = zeros((B, B), 'd')
    sumMU = 0.
    
    for j in range(N):
        mu = u[j,clas]
        sumMU  += mu
        w = training_set[j] - centers
        w =w[:,newaxis]
        sumX += dot(w, transpose(w))  * mu 
    cov = sumX  / float(sumMU)    
    return  cov    
  
def dis_GG(training_set,centers,fi,alf,deter,inv_fi):
    dif = (centers - training_set)[:,newaxis]
    dis = dot(transpose(dif),dot(inv_fi,dif))
    e = (float(dis)*0.5)
    distan=  (deter / alf) ** e
    return float(distan)      
        
