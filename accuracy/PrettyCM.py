import matplotlib.pyplot as plt
from numpy import array, sum, zeros, size


class statistical_analysis(object):

#Proposito:
#Definir la matriz de confusion y asi las diversas medidas estadisticas para medir la presicion de los clasificadores.  

        
        def __init__(self, y_true, y_pred, ncat):
                ''''
                Argumentos:
            
                -y_true = (numpy.ndarray):
                    Arreglo correspondeinte a los datos prosedente del mapa de verdad de terreno.
                    
                -y_pred (ndarray):
                    Arreglo  correspondiente al mapa de cobertura retornado por algun clasificador.
                
                -ncat (int):
                    Numero de categorias 
                '''
                self.ref= array(y_true)
                self.classification= array(y_pred)
                self.ncat =ncat+1
                self.confusion_matrix = self.confusion()
                self.prod_acc =self.producer_accuracy()
                self.user_acc =self.user_accuracy()
                self.reliability = self.reliability()
                self.kappa = self.coef_kappa()
        
        def confusion(self):
                ''''Retorna la matriz de confusion correspondiente.'''
                x =self.ref
                nx = size(x)
                y = self.classification
                nc =self.ncat 
                mc = zeros((nc, nc))
                for i in range(nx):
                        col = y[i]
                        fil = x[i]
                        mc[fil,col] = mc[fil,col]+1
                        
                return mc.astype(int)
        
        
        def producer_accuracy(self):
                ''''Retorna el arreglo correspondiente  a la precision del productor para cada categoria.'''
                nc =self.ncat   
                prod_acc = zeros(nc)
                mc = self.confusion_matrix
                for i in range(nc):
                        totalcols = sum(mc[:,i])
                        if  totalcols!=0 :
                                prod_acc[i] = float(mc[i,i])/float(totalcols)     
                        
                return prod_acc
                
                
        def user_accuracy(self):
                ''''Retorna el arreglo correspondiente  a la precision del usuario para cada categoria.'''
                nc =self.ncat   
                user_acc = zeros(nc)
                mc = self.confusion_matrix
                for i in range(nc):
                        totalrows = sum(mc[i,:])
                        if totalrows != 0 :
                                user_acc[i] = float(mc[i,i])/float(totalrows)             
                        
                return user_acc 
        
        def reliability(self):
                ''''Retorna el valor estadistico de fiabilidad.'''
                nc = self.ncat
                rel= 0.0
                x = self.ref
                n= size(x)
                mc = self.confusion_matrix
                for i in range(nc):
                        rel += mc[i,i]/float(n) 
                
                return rel      
                
        
        def coef_kappa(self):
                ''''Retorna el estadistico kappa.'''
                x = self.ref
                N = float(size(x))
                nc = self.ncat
                mc = self.confusion_matrix
                cols_rows =0.0
                totalrows = 0.0
                totalcols = 0.0
                diag = 0.0
                kappa=0.0
                for i in range(nc):
                        totalrows = sum(mc[i,:])
                        totalcols = sum(mc[:,i])
                        cols_rows += totalrows*totalcols
                        diag += mc[i,i]
                kappa = (N * diag - cols_rows)/ (N*N - cols_rows)
                
                return kappa
                                
                                
        def plot_mc(self):
                ''''ploteo de la matriz de confusion y de las diversas medidas estadisticas.'''
                conf_arr = self.confusion_matrix
                
                fig = plt.figure()
                plt.clf()
                ax = fig.add_subplot(221)
                ax.set_aspect(1)
                res = ax.imshow(array(conf_arr), cmap=plt.cm.jet, interpolation='nearest')
                nc = self.ncat
                width = len(conf_arr)
                height = len(conf_arr[0])
                for x in xrange(width):
                        for y in xrange(height):
                                ax.annotate(str(conf_arr[x][y]), xy=(y, x),horizontalalignment='center',
                                        verticalalignment='center')
                                        
                cb = fig.colorbar(res)
                plt.title('Matriz de Confusion') 
                plt.xlabel('Referencia') 
                plt.ylabel('Clasificacion') 
                alphabet = '0123-456789'
                alphabeto = '0123N456789'
                plt.xticks(range(width), alphabet[:width])
                plt.yticks(range(height), alphabeto[:height])
                
                cat = self.ncat
                filas =cat*2 + 2
                
                colors = [[(0.5,  1.0, 1.0) for c in range(1)] for r in range(2)]
                colors[0]= [(1., 0., 0.)]
                colors[1]= [(1., 0., 0.)]
                lightgrn = (0.5, 0.8, 0.5)
                etiquetas_fil1 = (u'Coeficiente kappa', u'Fiabilidad global')                                     
                etiquetas_fil = etiquetas_fil1[:filas]
                ax = fig.add_subplot(155 ,frameon=False, xticks=[], yticks=[]) 
                valores=[['%.4f' %(float(self.kappa))],['%.4f' %(self.reliability)]]                       

                plt.table(cellText=valores, rowLabels = etiquetas_fil,loc='upper center',cellColours=colors,rowColours=[lightgrn]*16)
                plt.savefig('confusion_matrix.png', format='png')
                return conf_arr
                
                        
                        
                
                
        
        
