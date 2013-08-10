'''
Various functions and algorithms for processing spectral data.
'''
# from numpy import ones

# import soft_classification.unsupervised.gkfcm as fkm
import numpy as np
# import color_list as co
from spectral import *
import spectral.io.envi as envi
import PIL
import Image
import os 
from soft_classification.supervised.FMLE import *
from soft_classification.unsupervised.GKFCM import *
from soft_classification.supervised.training_classes import *
from soft_classification.unsupervised.FCM import *
from soft_classification.unsupervised.WFCM import *
from soft_classification.unsupervised.GATH_GEVA import *
from soft_classification.unsupervised.PFCM import *
from soft_classification.unsupervised.PCM import *

from soft_classification.supervised.SFCM import *
from soft_classification.supervised.SWFCM import *
from soft_classification.supervised.SGKFCM import *

#from soft_classification.supervised.SFCMb import *
from soft_classification.accuracy.PrettyCM import *


from time import clock


def lookup_table():
    # Crea una lista de 3-tuples q son los posibles colores para la pos-clasificacion.
    black =     (0,0,0)
    white =     (255,255,255)
    red   =     (255,0,0)
    green =     (0,255,0)
    blue  =     (0,0,255)
    yellow=     (255,255,0)
    cyan  =     (0,255,255)
    magenta =   (255,0,255)
    maroon =    (176,48,96)
    seagreen =  (46,139,87)
    purple =    (160,32,240)
    coral =     (255,127,80)
    aquamarine =(127,255,212)
    orchid =    (218,112,214)
    sienna =    (160,82,45)
    chartreuse =(127,255,0)
    table = [black,blue,red,green,yellow,maroon,white,cyan,magenta,seagreen,purple,coral,aquamarine,orchid,sienna,chartreuse]
    return table

    # image_path="C:\\Users\\jose\\Desktop\\imagenes\\remanso"
        
def fuzzy_clusterings(k=3,type='e', emax=1.e-10, imax=15 ,me=None, m=2.,image_path='C:\\Users\\jose\\Desktop\\imagenes\\SAN-FLOAT', img_dest= "C:\\Users\\jose\\Desktop\\",verbose=False,win_size=3):
    '''
    Ejecucion del algoritmo fcm .
    Parameteros:
        image_path: ruta de la imagen a clasificar.
        img_dest=:  ruta donde se guardara la imagen.
    returns:
        imagen clasificada
    '''
    # crea un ImageArray object que contiene toda la interface de un numpy array apartir de un archivo de cabecera envi.
    img = create_image_from_envi_header(image_path)
    (nrows, ncols, nbands) = img.shape
    N = nrows * ncols
    if type != 'w':
        image = img.reshape((N, nbands))
    # crea una matriz membership de forma aleatoria
    if me :
        mu = me
    else:
        mu = random_fcm_membership(N,k)
    mu1 = np.random.sample((N,k))
    #crea la clase fuzzycmeas 
    if type == 'e':
        fcm = FuzzyCMeans(image, mu, 1.5,verbose=verbose)
    elif type == 'gus':
        fcm = GKFuzzyCMeans(image, mu, 1.5,verbose=verbose)
    elif type == 'gat':
        fcm = GGFuzzyCMeans(image, mu, 1.5,verbose=verbose)
    elif type == 'w':
        ws = win_size
        fcm = WFuzzyCMeans(img, mu, 1.5,verbose=verbose,win_size=ws)
    elif type == 'fp':
        fcm = PFCM(image, mu, 1.5,verbose=verbose)
    elif type == 'p':
        fcm = PCM(image, mu1, 1.5,verbose=verbose)
    tiempo_inicial = clock()
    #ejecuta la agrupacion en las distintas clases.
    fcm(emax=emax,imax=imax)
    tiempo_final = clock()
    membership = fcm.mu
    print membership.shape
    (N,C) = membership.shape
    mem = membership**2
    cp = 0.
    for i in range(C):
        cp =cp + np.sum(mem[:,i])
    cp=1./N * cp
    print "cp is :%f" %cp
    w = zeros((N,3))
    table_color = lookup_table()
    for k in range(N):
            maxi = argmax(membership[k,:])
            w[k,:] = table_color[maxi+1]
    w = w.reshape((nrows,ncols,3))
    
    if type == 'e':
        imagen = save_image(img_dest, "classified_image_by_fcm", w)
        print "runtime of FCM is: %f " % (tiempo_final - tiempo_inicial)
    elif type == 'gus':
        imagen = save_image(img_dest, "classified_image_by_Gustafon_Kessel", w)
        print "runtime of GUSTAFON_KESSEL is: %f " % (tiempo_final - tiempo_inicial)
    elif type == 'w':
        imagen = save_image(img_dest, "classified_image_by_weights", w)
        print "runtime of weights is: %f " % (tiempo_final - tiempo_inicial)
    return imagen
                
        
        
def create_image_from_envi_header(image_path):
    img = envi.open(image_path + '.hdr' , image_path)
    img = img.load()
    return img

    
    
    
   
def random_fcm_membership(nrsamples,nrclass):
    u =np.random.sample((nrsamples,nrclass))
    for j in range(nrsamples):              
        u[j,:]=u[j,:]/np.sum(u[j,:])
    return u
    
# def random_fcm_membership(nrsamples,nrclass):
    # u = (np.ones((nrsamples, nrclass)))*(1./nrclass)
    # return u     

def fmle(image_path="C:\\Users\\jose\\Desktop\\imagenes\\remanso", roi_path="C:\\Users\\jose\\Desktop\\imagenes\\remanso_map_verifi", gt_path="C:\\Users\\jose\\Desktop\\imagenes\\remanso_map_entre",img_dest= "C:\\Users\\jose\\Desktop\\",m=1.5):       
#def fmle(image_path="C:\\Users\\jose\\Desktop\\imagenes\\SAN-FLOAT", roi_path="C:\\Users\\jose\\Desktop\\imagenes\\ground_truth_sj",gt_path="C:\\Users\\jose\\Desktop\\imagenes\\roi_eval_sj", img_dest= "C:\\Users\\jose\\Desktop\\",m=1.5): 
    ''''
    Ejecucion del algoritmo fmle .
    Parameteros:
        image_path: ruta de la imagen a clasificar.
        roi_path=:  ruta de la roi de la imagen a clasificar.
        img_dest=:  ruta donde se desea guardar el mapa de clasificacion.
    '''
    from numpy import zeros, transpose, dot, newaxis,compress, indices, reshape, not_equal
    ## crea un ImageArray object que contiene toda la interface de un numpy array, apartir de la ruta de la roi .
    roi = envi.open( roi_path + '.hdr', roi_path).read_band(0)
    ## crea un ImageArray object que contiene toda la interface de un numpy array, apartir de un archivo de cabecera envi, que contien la imagen.
    img = envi.open( image_path + '.hdr', image_path).load()
    gt = envi.open( gt_path + '.hdr', gt_path).read_band(0)
    
    tiempo_inicial = clock()
    #crea las clases de entrenamiento
    classes = create_training_classes(img, gt,m=m)
    d = len(classes.classes)
    print d
    #crea el clasificador gaussiano
    gmlc = FuzzyGaussianClassifier(classes)
    #clasificacion de la imagen
    clMap = gmlc.classify_image(img)
    tiempo_final = clock()
    (nrows, ncols) = clMap.shape
    
    
    
    class_indices = set(roi.ravel())
    k = len(class_indices)-1
    print k
    
    class_indices = class_indices.difference([0])
    class_indices =list(class_indices)
    
    ind_clas ={}
    for i in range(k):
        ind_clas[class_indices[i]]=i
    
    (nrows, ncols) = roi.shape
    
    mask_index = numpy.not_equal(roi, 0.0)
       
    inds = transpose(indices((nrows, ncols)), (1, 2, 0))
    inds = reshape(inds, (nrows * ncols, 2))
    inds = compress(numpy.not_equal(mask_index.ravel(), 0), inds, 0).astype('h')
 
    ne= len(inds)
    roi_test = zeros(ne)
    img_test = zeros(ne)
    
    for i in range(inds.shape[0]):      
        x = roi[inds[i,0],inds[i,1]]
        roi_test[i]= ind_clas[x]
        y = clMap[inds[i][0], inds[i][1]]
        img_test[i]= int(y)
    

    
    w = np.zeros((nrows, ncols, 3))
    table_color = lookup_table()
    for i in range(nrows):
       for j in range(ncols):
            maxi = int(clMap[i,j])
            w[i,j,:] = table_color[maxi+1]
            if (maxi == k) :
                w[i,j,:] = table_color[0]
        
                
    imagen = save_image(img_dest, "classified_image_by_fmle", w)
    print "runtime of FMLE is: %f " % (tiempo_final - tiempo_inicial)
    
    return (roi_test,img_test) 

    
    
def save_image(img_dest_dir, filename, img):
    ''''
    Guarda una imagen en disco.
    Parameters:
            img_dest_dir: destino de la imagen.
            filename: nombre para el archivo.
            img: numpy matrix que contiene la imagen.
    '''
    name = filename
    if not filename:
        # crea nombre basado en el tiempo:
        time_format = '%d%m%y_%H%M%S'
        time_today = datetime.today()
        name = time_today.strftime(time_format)

    supported_extensions = ['png', 'jpeg']
    extension = 'png'
    filename = name + '.' + extension

    assert extension in supported_extensions, "ERROR: save_image(). " \
                                              "format not valid."
    # ruta de destino:
    img_dest_path = os.path.join(img_dest_dir, filename)

    # Convierte la imagen en numpy matrix usando  Image module:
    img_obj = Image.fromarray(img.astype(np.uint8))

    img_obj.save(img_dest_path, extension)
    img_obj = None

    print 'File saved to "' + img_dest_path + '".'
    
#def super(type='e', emax=1.e-10, imax=150,m=2.,image_path="C:\\Users\\jose\\Desktop\\imagenes\\SAN-FLOAT", gt_path="C:\\Users\\jose\\Desktop\\imagenes\\ground_truth_sj", roi_path="C:\\Users\\jose\\Desktop\\imagenes\\roi_eval_sj",img_dest="C:\\Users\\jose\\Desktop\\",win_size = 3):   
def super(type='e',emax=1.e-10, imax=150,m=2.,image_path="C:\\Users\\jose\\Desktop\\imagenes\\remanso", roi_path="C:\\Users\\jose\\Desktop\\imagenes\\remanso_map_verifi", gt_path="C:\\Users\\jose\\Desktop\\imagenes\\remanso_map_entre",img_dest="C:\\Users\\jose\\Desktop\\",win_size = 3):
    from numpy import zeros, transpose, dot, newaxis,compress, indices, reshape, not_equal,amax

    img = create_image_from_envi_header(image_path)
    roi = envi.open( roi_path + '.hdr', roi_path).read_band(0)
    gt = envi.open( gt_path + '.hdr', gt_path).read_band(0)
    
    
    
    
    (nrows, ncols, nbands) = img.shape
    N = nrows * ncols
    #mu = random_fcm_membership(N,3)
    # if win_size:
        # win = win_size
    
    if type == 'e':
        fcm = SupFuzzyCMeans(img,gt,m,False)
    elif type == 'gus':
        fcm = SupGKFuzzyCMeans(img,gt,m,False)
    elif type == 'w':
        fcm = SupWFuzzyCMeans(img,gt,m,False,win_size=win_size)
    
    tiempo_inicial = clock()
    fcm(emax=emax,imax=imax)
    tiempo_final = clock()

    
    membership = fcm.mu
    des=fcm.desvs
    me=fcm.means
    ind_clas =fcm.index_class
    (_,nclas) = membership.shape #len(ind_clas)
    print nclas
    
    cen=fcm.c 
    count=0
    print membership
    w = zeros(N)
    classified = zeros((N,3))
    table_color = lookup_table()
    for k in range(N):
            maxi = argmax(membership[k,:])
            w[k] = maxi
            classified[k,:]=table_color[maxi+1]
            if amax(membership[k,:]) < 0.51:
                count = count + 1
                w[k] = int(nclas)
                classified[k,:]=table_color[0]
            
    w = w.reshape((nrows,ncols))
    classified = classified.reshape((nrows,ncols,3))
    
    mask_index = numpy.not_equal(roi, 0.0)
     
    inds = transpose(indices((nrows, ncols)), (1, 2, 0))
    inds = reshape(inds, (nrows * ncols, 2))
    inds = compress(numpy.not_equal(mask_index.ravel(), 0), inds, 0).astype('h')
    
    ne= len(inds)
    roi_test = zeros(ne)
    img_test = zeros(ne)
    co=0
    for i in range(inds.shape[0]):      
        x = roi[inds[i,0],inds[i,1]]
        roi_test[i]= ind_clas[x]
        y = w[inds[i][0], inds[i][1]]
        img_test[i]= int(y)
        if y == 4:
            co=co+1
    print co     
        
    
    
    if type == 'e':
        imagen = save_image(img_dest, "classified_image_by_SFCM", classified)
        print "runtime of FCM is: %f " % (tiempo_final - tiempo_inicial)
    elif type == 'gus':
        imagen = save_image(img_dest, "classified_image_by_SGK", classified)
        print "runtime of GUSTAFON_KESSEL is: %f " % (tiempo_final - tiempo_inicial)
    elif type == 'w':
        imagen = save_image(img_dest, "classified_image_by_SWFCM", classified)
        print "runtime of WFCM is: %f " % (tiempo_final - tiempo_inicial)    
    
    return (img_test,roi_test,count)
    
    




















