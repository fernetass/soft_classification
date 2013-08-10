import numpy
from soft_classification.utils.distances  import distance
from numpy import dot, array, sum, zeros, outer, any, apply_along_axis,argmax,ones,newaxis,zeros_like,transpose,amax,sqrt

from FCM import FuzzyCMeans
from time import clock




def euclidean(a,b):
    #Calculates the euclidian distance between two data points.     
    result = (a-b)**2
    return float(sqrt(sum(result)))

def dot_density_function(img, win_size=3):
    """
    Funcion de densidad para cada pixel de la imagen
    Parametros:
        - img: numpy matrix  que representa la imagen.
        - win_size: tamanano de la ventana
    """
    assert win_size >= 3, 'ERROR: win size must be at least 3'
    assert win_size % 2 != 0, 'ERROR: win size must be odd'
    
    (nrows, ncols, nbands) = img.shape
    win_offset = win_size / 2
    Z =zeros((nrows, ncols))
    
    a=sqrt(nbands)*255.
    
    for i in xrange(0, nrows):
        xleft = i - win_offset
        xright = i + win_offset

        if xleft < 0:
            xleft = 0
        if xright >= nrows:
            xright = nrows

        for j in xrange(0, ncols):
            yup = j - win_offset
            ydown = j + win_offset

            if yup < 0:
                yup = 0
            if ydown >= ncols:
                ydown = ncols
                
            sumZ =0.
            for x in xrange(xleft,xright):
                for y in xrange(yup,ydown):
                    sumZ += 1./ (euclidean(img[x,y,:],img[i,j,:])+a)
            Z[i, j] = sumZ


    return Z


    
def weights_matrix(img,win_size):       
    (nrows, ncols, nbands) = img.shape
    M = nrows*ncols
    w =zeros((M))
    tiempo_inicial = clock()
  
    z = dot_density_function(img, win_size)
    z = z.reshape((nrows*ncols))
    sumz= sum(z)    
    for i in range(M):
        w[i]= float(z[i])/ sumz
    tiempo_final = clock()
    print "runtime of construction weight matrix: %f " % (tiempo_final - tiempo_inicial)
    return w


################################################################################
# Fuzzy C-Means class
################################################################################
class WFuzzyCMeans(FuzzyCMeans):
    '''
    Esta clase es usada para instanciar wfuzzy c-means object.
    Para instanicar el objeto se requiere un conjunto de datos de entrenamiento y un conjunto de condicones iniciales.
    El conjunto de datos de entrenamiento es una lista o array de vectores  N-dimensionales.
    El conjunto de condiciones iniciales son una lista de los membership values iniciales,para cada vector de los datos de entrenamiento.
    La longitud de ambas lista debera ser iguales.the
    El numero de columnas en la lista de conjunto de condiciones inciales deber ser igual al numero de clases.
    

    Hay restricciones en las condiciones iniciales: en primer lugar, ninguna columna puede ser
    ceros o unos - si eso ocurriera, entonces la clase descrita por esta columna es innecesaria, en segundo lugar,
    la suma de las pertenencias de cada dato debe ser uno.Esto significa que la condicion inicial es una
    particion especia de C subconjuntos.
    '''
    
    def __init__(self, training_set, initial_conditions, m=2.,verbose=False, win_size=5):
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
       # img = training_set
        self.weights = weights_matrix(training_set,win_size)
        (nrows, ncols, nbands) = training_set.shape
        N = nrows * ncols
        training_set = training_set.reshape((N, nbands))
        
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
        x = self.x
        w = self.weights
        M, N = x.shape
        mm = self.mu ** self.m 
        sumd = dot(w.T,mm)
        
        prod = zeros((M,N))
        for i in range(M):
            prod[i] = x[i] * w[i]
            
        c = dot(prod.T, mm) / sumd
        self.c = c.T
        return self.c
        
        
        





