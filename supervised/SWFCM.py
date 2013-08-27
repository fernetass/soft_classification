'''
Implementacion del metodo de clasificacion supervisado Weighted Fuzzy C-Means.
'''

import numpy
from numpy import dot, array, sum, zeros, outer, any, apply_along_axis,argmax,ones,newaxis,zeros_like,transpose,amax,sqrt
from soft_classification.unsupervised.WFCM  import WFuzzyCMeans,euclidean,dot_density_function,weights_matrix
from SFCM import SupFuzzyCMeans,mean_std,calc_stats_class
from time import clock






################################################################################
#Supervised Weighted Fuzzy C-Means class
################################################################################
class SupWFuzzyCMeans(WFuzzyCMeans):
    '''
        Clase derivada de clase WFuzzyCmeans, es usada para instanciar el metodo WFCM supervisado.
    '''
    
    def __init__(self, training_set, class_mask, m=2.,verbose=False, win_size=5):
        '''
        Inicializa el algoritmo.

        Parameteros:
        training_set
            lista o array  de vectores que contienen los datos a ser clasificados.
            Cada uno de los vectores en esta lista deben tener la misma dimension,
            o el algoritmo no se comportara correctamente.
            
        initial_conditions
            lista o array de vectores que contienen loso membership values inciales asociados 
            a cada ejemplo de los datos del conjunto de entrenamiento. 
            Cada columna de este array contiene los membership asignados a las correspondientes clases para cada dato de entrenamiento.
            
        m
            peso de pertenencia (fuzzyness coefficient). 
            Deber ser > 1 .Cuando es = 1 se establece como una particion dura, por defecto el valor es 2.
            
        iter 
            Numero que estable en que iteracion se encuentra el algoritmo.
            
        verbose 
            Valor booleano para establecer si el algoritmo imprimira datos relativos al avance del mismo.
        win_size  
            Entero que estable el tamano de la ventana para la funcion de asignacion de pesos (weights_matrix).
        '''
        (self.means, self.desvs, self.index_class) = mean_std(training_set,class_mask)
        self.weights = weights_matrix(training_set,win_size)
        (nrows, ncols, nbands) = training_set.shape
        N = nrows * ncols
        training_set = training_set.reshape((N, nbands))
        self.iter = 0
        self.m = m
        self.verbose = verbose
        self.x = array(training_set)
        self.mu = self.membership()
        
        
        self.c = self.centers()
        
        
                
        

    def membership(self):
        '''
        Dato el presente estado del algoritmo, recalcula los membership de cada dato sobre cada clase.
        Es decir, modifica las condiciones iniciales para representar un nivel de evolucion del algoritmo. 
        Observar que este metodo modifica el estado del algoritmo, si cualquier cambio fue realizado sobre algun parametro.

        Returns
          Vector  que contiene en cada fila los membership de los correspondientes datos en cada clase.
        '''
        
        from spectral import status
        x = self.x
        i = self.iter
        if i==0 :
            c = self.means
            self.iter += 1 
        else: 
            c = self.c
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





