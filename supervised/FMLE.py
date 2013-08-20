import numpy

class Classifier:
    '''
    Clase base para un clasificador.
    '''
    def __init__(self):
        pass
        
    def classify_image(self, image):
        '''Clasifica una imagen completa, devolviendo un mapa de clasificacion.      
        Argumentos:
            -image: ndarray de dimension `MxNxB` que corresponde a la imagen a clasificar.     
        Returns (ndarray):
            -class_map: ndarray de dimension `MxN` de enteros  correspondiente al mapa de clasificasion.
        '''
    
        from spectral import status
        from training_classes import ImageIterator
        from numpy import zeros
        status.display_percentage('Classifying image...')
        it = ImageIterator(image)
        class_map = zeros(image.shape[:2])
        N = it.get_num_elements()
        i, inc = (0, N / 100)
        for spectrum in it:
            class_map[it.row, it.col] = self.classify_spectrum(spectrum)
            i += 1
            if not i % inc:
                status.update_percentage(float(i) / N * 100.)
        status.end_percentage()
        return class_map

            
        
        
class FuzzyGaussianClassifier(Classifier):
    # Clasificador fuzzy Maximum Likelihood 
    def __init__(self, training_data = None, min_samples = None):
        '''Crea el clasificador y opcionalmente un conjunto de datos de entrenamiento.
        
        Argumentos:
            -training_data: clase (TrainingClassSet) correspondiente al conjunto de datos de entrenamiento [default None].
            -min_samples: (int) correspondiente al  Minimo numeros de datos requeridos en las clases de entrenamiento
                          para ser includa en el clasificador [default None].
                
        '''
        if min_samples:
            self.min_samples = min_samples
        else:
            self.min_samples = None
        if training_data:
            self.train(training_data)

    def train(self, training_data):
        '''
            Argumento:
                -training_data: clase (TrainingClassSet)
        '''
        if not self.min_samples:
            # establece el minimo numero de muestras de como el numero de bandas de la imagen.
            self.min_samples = training_data.nbands
        self.classes = []
        for cl in training_data:
            if cl.size() >= self.min_samples:
                self.classes.append(cl)
            else:
                print '  Omitting class %3d : only %d samples present' % (cl.index, cl.size())
        for cl in self.classes:
            if not hasattr(cl, 'stats'):
                cl.calc_stats()


    def classify_spectrum(self, x):
        '''
        Clasifica un pixel dentro de una clase de entrenamiento.
        Argumentos:
            -x : ndarray correspondiente al pixel a clasificar. 
        Returns:
            -classIndex: (int) correspondiente a la clase asignada al pixel x.
        '''
        from numpy import dot, transpose, zeros,argmax, sqrt, absolute, exp ,pi
        from numpy.oldnumeric import NewAxis
        from math import log
        from numpy.linalg import det,inv
        max_prob = -100000000000.
        max_class = -1
        first = True
        d = len(self.classes)
        B=x.shape[0]
        probs= zeros(d)
        degree=zeros(d)
        indexs=zeros(d)
        j=0
        for cl in self.classes:
            delta = (x - cl.stats.mean)[:, NewAxis]
            i = 1.0 / (((2*pi)**(B/2.))* (sqrt(absolute(det(cl.stats.cov)))))
            e = exp((- 0.5) * dot(transpose(delta), dot(cl.stats.invCov, delta)))
            prob = i * e
            probs[j] = prob[0,0]
            indexs[j]=cl.index
            j=j+1
        sum =0.0
        for z in range(j):
            sum+= probs[z]
        for q in range(d):
            degree[q] =(probs[q])/sum   
        ind=argmax(degree)
        max_class= argmax(degree)
        if degree[ind] < 0.75:
           max_class= d
        return max_class


    
    


