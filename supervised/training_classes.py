import numpy
import spectral.io.envi as envi



class ImageIterator():
    '''
    Iterador sobre todos los pixeles de una imagen.
    '''
    def __init__(self, im):
        self.image = im
        self.numElements = im.shape[0] * im.shape[1]
    def get_num_elements(self):
        return self.numElements
    def get_num_bands(self):
        return self.image.shape[2]
    def __iter__(self):
        from spectral import status
        (M, N) = self.image.shape[:2]
        count = 0
        for i in range(M):
            self.row = i
            for j in range(N):
                self.col = j
                yield self.image[i, j]



def mean_cov(image, mask = None, index = None, u=None, ind_clas=None):
    '''
    Calculo de la media y covarianza difusa de un conjunto de vectores.
    Argumentos:
        -image (ndarrray):
            matriz `MxNxB` o clase spectral.Image procedente de la libreria Spectral Python
        -mask (ndarray):        
            Si `mask` se especifica, la media y covarianza se calculan para todos
            pixeles indicado en la matriz  mask. Si `index` se especifica, todos
            pixeles de la imagen para los cuales `mask == index` seran usados, en caso contrario,
            todos los elementos distintos de cero de mask se utilizaran [default None].     
        -index (int):       
            Especifica el valor en 'mask' que se utilizara para seleccionar los pixeles de 'image'.
            Si no se especifica pero 'mask' esta, entonces todos los elementos no nulos de 'mask'
            se utilizaran.
            Si ni 'mask' ni `index` se especifican: 'todas las muestras de vectores se utilizaran [default None].       
        -u (ndarray):
            matriz  membership          
        -ind_clas (dic):
            Establece la asociacion de los distintos 'indexs' a sus categorias [default None].
    Returns 
        3-tuple  que contiene:
        -mean(ndarray): media
        -cov(ndarray):covarianza difusa.
        -count(int): numero de muestras usados para el calculo de la media y covarianza
    '''
    import spectral
    from spectral import status
    from numpy import zeros, transpose, dot, newaxis,compress, indices, reshape, not_equal,cov
    from spectral.io import typecode
    
    typechar = typecode(image)
    (nrows, ncols, B) = image.shape
    (nr,nc) = mask.shape
    mask_i = numpy.equal(mask, index)
    mask_array = mask.reshape(nr*nc)
    mask_index = numpy.equal(mask_array, index)
    nSamples = sum(mask_index.ravel())
    
    clas = ind_clas[index]
    
    inds = transpose(indices((nrows, ncols)), (1, 2, 0))
    inds = reshape(inds, (nrows * ncols, 2))
    inds = compress(not_equal(mask_i.ravel(), 0), inds, 0).astype('h')

   # sumX = zeros((B,), 'd')
    sumX_MU = zeros((B,), 'd')
    sumX2 = zeros((B, B), 'd')
    vector=zeros((inds.shape[0], B), 'd')
    count = 0
    count1 = 0
    sumMU = 0
    
    statusInterval = max(1, nSamples / 100)
    status.display_percentage('Covariance.....')
    for i in range(inds.shape[0]):
        x = image[inds[i][0], inds[i][1]]
        mu = u[inds[i][0], inds[i][1],clas]
        xmu = x * mu
        count += 1
        sumMU  += mu
        sumX_MU += xmu
 #       vector[i]=x
           
    mean = sumX_MU / sumMU
    print vector.shape
    #cova=cov(transpose(vector))
    for j in range(inds.shape[0]):
        count1 += 1
        sample = image[inds[j][0], inds[j][1]]
        mu = u[inds[j][0], inds[j][1],clas]
        w = sample - mean
        w =w[:,newaxis]
        sumX2 += dot(w, transpose(w))  * mu 
        if not count1 % statusInterval:
                status.update_percentage(float(count1) / nSamples * 100.)   
    cova = sumX2  / float(sumMU)
    status.end_percentage()
    return (mean, cova, count)

    

class GaussianStats:
    def __init__(self):
        self.nsamples = 0

class TrainingClass:
    def __init__(self, image, u, ind_clas, mask, index = 0, class_prob = 1.0 ):
        '''Crea una nueva clase de entrenamiento que se define mediante la aplicacion de mask en `image`.
        
    Arguments:
    
        -image (clase spectral.Image o numpy.ndarray):
            imagen `MxNxB` sobre la que se define la clase de entrenamiento.
            
        -u (ndarray):
            matriz de membresia (membership)
        
        -ind_clas (dic):
            Establece la asociacion de los distintos 'indexs' a sus categorias.
            
        -mask(numpy.ndarray):
            Una matriz MxN de enteros que indica como se asocian loos pixeles de`image`  con las clases.
        
        -index (int) [default 0]:
            si `index` == 0, todos los elementos no nulos de 'mask' se asocian con la clase. 
            Si `index` es distinto de cero, todos los elementos de 'mask' igual a `index' se asocian con la clase.

        -classProb (float) [default 1.0]:
            Define la probabilidad a priori asociada con la clase. 
        '''
        self.image = image
        self.nbands = image.shape[2]
        self.mask = mask
        self.index = index
        self.class_prob = class_prob
        self.mu = u
        self.ind_clas = ind_clas
        self._stats_valid = 0
        self._size = 0

    def size(self):
        '''Retorna el numero de pixeles en training set.'''
        from numpy import sum, equal
        if self._stats_valid:
            return self._size

        if self.index:
            return sum(equal(self.mask, self.index).ravel())
        else:
            return sum(not_equal(self.mask, 0).ravel())        

    def calc_stats(self):
        '''
        Calcula las estadistica de la clase.
    
        Este metodo se usa para actulizar el atributo `stats` de la clase.
        Donde `stats` tiene los siguientes subatributo:
        
        =============  ======================   ===================================
        Atributo         Tipo                         Descripcion
        =============  ======================   ===================================
        `mean`         :clase:`numpy.ndarray`   B-vector de media
        `cov`          :clase:`numpy.ndarray`   `BxB` matriz de covarianza difusa
        `inv_cov`      :clase:`numpy.ndarray`   Inversa de cov
        =============  ======================   ===================================
        '''
        import math
        from numpy.linalg import inv, det

        self.stats = GaussianStats()
        (self.stats.mean, self.stats.cov, self.stats.nsamples) = \
                          mean_cov(self.image, self.mask, self.index,self.mu,self.ind_clas)
        self.stats.invCov = inv(self.stats.cov)
        self._size = self.stats.nsamples
        self._stats_valid = 1




            
class TrainingClassSet(object):
    '''  manejo de un conjunto de clases 'TrainingClass'''
    def __init__(self):
        self.classes = {}
        self.nbands = None  
    def __getitem__(self, i):
        '''Retorna la clase de entrenamiento con ID i.'''
        return self.classes[i]
    def __len__(self):
        '''Retorna el numero de clases de entrenamieto que contiene el conjunto.'''
        return len(self.classes)
    def add_class(self, cl):
        '''Agrega una nueva clase al conjunto training set.
    
        Argumentos:
            -cl (clase` TrainingClass`):
        '''
        if self.classes.has_key(cl.index):
            ''''`cl.index` No debe duplicarse una clase que ya esta en el conjunto.'''
            raise Exception('Attempting to add class with duplicate index.')
        self.classes[cl.index] = cl
        if not self.nbands:
            self.nbands = cl.nbands
                
    def __iter__(self):
        '''
        Retorna un iterador para todas las clases `TrainingClass` en el conjunto.
        '''
        for cl in self.classes.values():
            yield cl
    

    
  

def create_training_classes(image, class_mask, calc_stats = 0, m=1.5):
    '''
    Crea una clase TrainingClassSet .
    Argumentos:   
        -image (clase `spectral.Image` o numpy.ndarray)       
        -class_mask (numpy.ndarray):
            si `class_mask [i, j] ==` `k`, el pixel `image[i, j] se supone que pertenece a las clase k.      
        -calc_stats:
            Un parametro opcional el cual, de ser cierto, hace que las estadisticas sean calculadas 
            para todas las clases de entrenamiento.
        m :peso de pertenencia (fuzzyness coefficient).             
    Returns:
        TrainingClassSet object.
        
    Las dimensiones de classMask debe ser la misma que las dos primeras dimensiones
    de la imagen correspondiente. Los valores cero en classMask se consideran
    sin etiqueta y no se agregan a ningun conjunto de entrenamiento.
    '''
    import numpy as np
    class_indices = set(class_mask.ravel())
    k = len(class_indices)-1
    
    class_indices = class_indices.difference([0])
    class_indices =list(class_indices)
    
    ind_clas ={}
    for i in range(k):
        ind_clas[class_indices[i]]=i
    
    
    (nrows, ncols, nbands) = image.shape
    N = nrows * ncols
    U = np.random.sample((nrows,ncols,k))
    for i in range(nrows):
        for j in range(nrows):
            U[i,j,:]=U[i,j,:]/np.sum(U[i,j,:])
    
    UU= U ** m
    class_indices = set(class_mask.ravel())
    classes = TrainingClassSet()
    for i in class_indices:
        if i == 0:
            continue
        cl = TrainingClass(image, U,ind_clas, class_mask, i)
        if calc_stats:
            cl.calc_stats()
        classes.add_class(cl)
    return classes


