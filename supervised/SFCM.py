import numpy
from soft_classification.unsupervised.FCM  import FuzzyCMeans
from numpy import dot, array, sum, zeros, outer, any, apply_along_axis,argmax,ones,newaxis,zeros_like,transpose,sqrt




def mean_std(training_set,class_mask):
    '''calculo de los vectores de media y varianza.

        Argumentos:   
        -training_set(numpy.ndarray):
            lista o array  de vectores que contienen los datos a ser clasificados.
        -class_mask (numpy.ndarray):
            si `class_mask [i, j] ==` `k`, el pixel `image[i, j] se supone que pertenece a las clase k.      
        Returns 
            3-tuple  que contiene:
            -medias(ndarray): media de cada categoria.
            -desv(ndarray): desviacion estandar de cada categoria.
            -ind_clas(dic): asociacion de los distintos 'indexs' a sus categorias.
    '''
    import spectral
    from numpy import zeros
    
    (nrows, ncols, B) = training_set.shape
    class_indices = set(class_mask.ravel())
    k = len(class_indices)-1
    
    class_indices = class_indices.difference([0])
    class_indices = list(class_indices)
    
    medias = zeros((k, B), 'd')
    desv = zeros((k, B), 'd')
    i=0
    ind_clas = {}
    for j in class_indices:
        ind_clas[class_indices[i]]=i    
        (medias[i],desv[i]) = calc_stats_class(training_set,class_mask,j)
        i=i+1
    return(medias,desv,ind_clas)
        
        
    
def calc_stats_class(image, mask = None, index = None):
    ''' funcion auxiliar para el calculo estadistico de cada categoria.'''
    import spectral
    from numpy import zeros, transpose,compress, indices, reshape, not_equal,mean,std
    from spectral.io import typecode
    
    typechar = typecode(image)
    (nrows, ncols, B) = image.shape
    (nr,nc) = mask.shape
    mask_i = numpy.equal(mask, index)
    mask_array = mask.reshape(nr*nc)
    mask_index = numpy.equal(mask_array, index)
    nSamples = sum(mask_index.ravel())
    
    
    inds = transpose(indices((nrows, ncols)), (1, 2, 0))
    inds = reshape(inds, (nrows * ncols, 2))
    inds = compress(not_equal(mask_i.ravel(), 0), inds, 0).astype('h')

    vector=zeros((inds.shape[0], B), 'd')
 
    for i in range(inds.shape[0]):
        x = image[inds[i][0], inds[i][1]]
        vector[i] = x     
    media = mean(vector,axis=0)
    desv = std(vector, axis=0)
    return(media,desv)









################################################################################
# Supervised Fuzzy C-Means class
################################################################################
class SupFuzzyCMeans(FuzzyCMeans):
    '''
    Clase derivada de clase base FuzzyCmeans, es usada para instanciar el metodo FCM de naturaleza supervisada.
    '''
    
    def __init__(self, training_set,class_mask, m=2.,verbose=False):
        '''
        Inicializa el algoritmo.

        Parameters
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
            
        '''
        
        (self.means, self.desvs, self.index_class) = mean_std(training_set,class_mask)
        (nrows, ncols, nbands) = training_set.shape
        N = nrows * ncols
        training_set = training_set.reshape((N, nbands))
        self.m = m
        self.verbose = verbose
        self.x = array(training_set)
 
        self.iter = 0
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



