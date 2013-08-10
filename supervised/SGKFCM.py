################################################################################
import numpy
from soft_classification.utils.distances  import distance
from numpy import dot, array, sum, zeros, outer, any, apply_along_axis,argmax,ones,newaxis,zeros_like,transpose
from numpy.linalg import det,inv
from SFCM import SupFuzzyCMeans,mean_std,calc_stats_class


################################################################################
# GK -FuzzyCmeans class
################################################################################
class SupGKFuzzyCMeans(SupFuzzyCMeans):
    def __init__(self, training_set,class_mask, m=2.,verbose=False):
        
        (self.means, self.desvs, self.index_class) = mean_std(training_set,class_mask)
        (nrows, ncols, nbands) = training_set.shape
        N = nrows * ncols
        training_set = training_set.reshape((N, nbands))
        self.m = m
        self.verbose = verbose
        self.x = array(training_set)
        self.mu = self.init_mem()
        self.iter = 0
        self.c = self.centers()
        
        
    def init_mem(self):
        '''
        Dato el presente estado del algoritmo, recalcula los membership de cada dato sobre cada clase.
        Es decir, modifica las condiciones iniciales para representar un nivel de evolucion del algoritmo. 
        Observar que este metodo modifica el estado del algoritmo, si cualquier cambio fue realizado sobre algun parametro.

        Returns
          Vector  que contiene en cada fila los membership de los correspondientes datos en cada clase.
        '''
        
        from spectral import status
        x = self.x
        c = self.means
        M, _ = x.shape
        C, _ = c.shape
        r = zeros((M, C))
        m1 = 1./(self.m-1.)
        if self.verbose:
            print 'iteration number %d ' % self.iter
            status.display_percentage('updating membership...')

        for k in range(M):
            den = sum((x[k] - c)**2., axis=1)
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

    def membership(self):

        from spectral import status
        x = self.x
        c = self.c
        (M, B) = x.shape
        C, _ = c.shape
        r = zeros((M, C))
        m1 = 2./(self.m-1.)
        if self.verbose:
            print 'iteration number %d ' % self.iter
            status.display_percentage('updating membership...')
        Ai = ones((C,B,B))    
        fi = zeros((C,B,B))
        for i in range(C):
            fi[i] = covariance_matrix(self.x,self.mu,self.c[i],self.m,i)
            Ai[i] = dot(det(fi[i]) ** (1./6), inv(fi[i]))
        den = zeros((C)) 
        for k in range(M):
            for j in range(C):
                den[j] = dis_GK(self.x[k],self.c[j],Ai[j])
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
    
    
       
def dis_GK(training_set,centers,Ai):
    dif = (training_set - centers)[:,newaxis]
    dis = dot(transpose(dif),dot(Ai,dif))
    return float(dis)      
        
