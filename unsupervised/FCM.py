'''
Implementacion del metodo de clustering Fuzzy C-means.
'''
import numpy
from numpy import dot, array, sum, zeros, outer, any, apply_along_axis,argmax,ones,newaxis,zeros_like,transpose,sqrt



################################################################################
# Fuzzy C-Means class
################################################################################
class FuzzyCMeans(object):
    '''
    Esta clase es usada para instanciar fuzzy c-means object.
    Para instanicar el objeto se requiere un conjunto de datos de entrenamiento y un conjunto de condicones iniciales.
    El conjunto de datos de entrenamiento es una lista o array de vectores  N-dimensionales.
    El conjunto de condiciones iniciales son una lista de los membership values iniciales (matriz de pertenencias),para cada vector de los datos de entrenamiento.
    La longitud de ambas lista debera ser iguales.
    El numero de columnas en la lista de conjunto de condiciones inciales deber ser igual al numero de clases.
    

    Hay restricciones en las condiciones iniciales: en primer lugar, ninguna columna puede ser
    ceros o unos - si eso ocurriera, entonces la clase descrita por esta columna es innecesaria, en segundo lugar,
    la suma de las pertenencias de cada dato debe ser uno.Esto significa que la condicion inicial es una
    particion especia de C subconjuntos.
    '''
    
    def __init__(self, training_set, initial_conditions, m=2.,verbose=False):
        '''
        Inicializa el algoritmo.

        Parameters
        training_set
            lista o array  de vectores que contienen los datos a ser clasificados.
            Cada uno de los vectores en esta lista deben tener la misma dimension,
            o el algoritmo no se comportara correctamente.
            
        initial_conditions
            lista o array de vectores que contienen los membership values inciales asociados 
            a cada dato del conjunto de entrenamiento. 
            Cada columna de este array contiene los membership asignados a las correspondientes clases para cada dato de entrenamiento.
            
        m
            peso de pertenencia (fuzzyness coefficient). 
            Deber ser > 1 .Cuando es = 1 se establece como una particion dura, por defecto el valor es 2.
            
        iter 
            Numero que estable en que iteracion se encuentra el algoritmo.
            
        verbose 
            Valor booleano para establecer si el algoritmo imprimira datos relativos al avance del mismo.
            
        '''
        
        self.x = array(training_set)
        self.mu = array(initial_conditions)
        self.m = m
        self.c = self.centers()
        self.iter = 0
        self.verbose = verbose

    def centers(self):
        '''
        Dato el presente estado del algoritmo, recalcula los centroides, es decir,
        los vectores que representan cada una de las clases. Observar que este metodo modifica el estado del algoritmo,
        si cualquier cambio fue realizado sobre algun parametro.

        Returns
          Vector que contiene en cada fila los centroides del algoritmo.
        '''
        mm = self.mu ** self.m
        c = dot(self.x.T, mm) / sum(mm, axis=0)
        self.c = c.T
        return self.c

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

    def step(self):
        '''
        Este metodo ejecuta un paso del algoritmo. 
        
        Returns
            la diferncia entre las ultimas 2 matrices de centroides. 
            Se puede utilizar para realizar un seguimiento de convergencia y como una estimacion del error.
        '''
       
        old = self.c
        self.membership()
        self.centers()
        self.iter += 1 
        return sqrt(sum((self.c - old)**2.))

    def __call__(self, emax=1.e-10, imax=20):
        '''
         ``__call__`` Es la interface usada para correr el algoritmo hasta q la convergencia se establezca.

        Parametros
        emax
            Especifica el maximo error admitido en la ejecucion del algoritmo.Por defecto es de 1.e-10. 
        imax
            Especifica el numero maximo iteraciones admitido en la ejecucion del algoritmo.

        Returns
          array que contiene en cada fila, los vectores representantes de los centros de los cluster (centroides).
        '''
        error = 1.
        i = 0
        while error > emax and i < imax:
            print 'iteration nr :',i
            error = self.step()
            print error
            i = i + 1
        return self.c



