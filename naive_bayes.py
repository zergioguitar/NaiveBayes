############################################
###            Desarrollado por          ###
###       Sergio Elías Flores Labra      ###
###         serflo.elec@gmail.com        ###
############################################
###        Optimizado para Python 3		 ###
############################################
###          Requisitos:				 ###
###         	numpy					 ###
###         	tkinter					 ###
############################################

#######  Imports  #######
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#######  Class  #######
class NaiveBayes( ):
	def __init__( self ):
		self.train = []
		self.result = {}

	def train_model( self, train ):
		self.train = train
		tipos = clases( self.train )
		self.result = resumen( tipos )
		return self.result

	def probability( self , test ):
		respuesta=[]
		for j in range(len(test)): #para cada instancia de testeo
			probabilities = {}
			for tipo, instancias in self.result.items():
				probabilities[tipo] = 1
				for i in range(len(instancias)):
					avg, std = instancias[i]
					x = float(test[j][i])
					probabilities[tipo] *= gaussian(x, avg, std)
			respuesta.append(probabilities)
		return respuesta

	def predict( self, predictions):
		result = []
		for i in range(len(predictions)):
			prediction = predictions[i]
			bestLabel = None
			bestProb = -10000.0
			for tipo, probability in prediction.items():
				if float(probability) > float(bestProb):
					bestProb = probability
					bestLabel = tipo
			result.append([bestLabel,bestProb])
		return result

	def graph(self):
		for tipo, instancias in self.result.items():
			#solo grafica el primer atributo de los datos.
			mu= instancias[0][0]
			variance= instancias[0][1]
			sigma = math.sqrt(variance)
			x = np.linspace(mu-3*variance,mu+3*variance, 100)
			y = mlab.normpdf(x, mu, sigma)
			plt.plot(x,y)
			plt.annotate(tipo, xy=(x[50], y[50]))
		plt.show()

#######  Functions  #######

# Metodo que genera un resumen de las instancias, 
# separando por clases, y entregando el promedio 
# y varianza de c/u de las instancias
def resumen( data ):
	aux = {}
	for tipo,instancias in data.items():
		avg = average( instancias )
		std = stdev( instancias , avg )
		result = []
		for x in range(len(avg)):
			result.append([avg[x], std[x]])
		aux[tipo] = result
	return aux
	
# Metodo que retorna las clases de los datos
def clases( data ):
	tipos = {}
	for i in range( len(data) ):
		instancia = data[i]
		if( instancia[-1] not in tipos ):
			tipos[instancia[-1]]= []
		tipos[instancia[-1]].append(instancia)
	return tipos

# Metodo que calcula el promedio de los datos.
def average( data ):
	prom = [] 
	columns= len(data[0])-1
	rows = len(data)
	for i in range( columns ):
		aux = 0.0
		for j in range( rows ):
			aux+= float(data[j][i])
		prom.append( float(aux/(rows)) )
	return prom

# Metodo que calcula la desviacion estandar de los datos.
def stdev( data , avg ):
	varianza = []
	columns= len(data[0])-1
	rows = len(data)
	for i in range( columns ):
		aux = 0.0
		for j in range( rows ):
			aux+= pow( float(data[j][i])-avg[i] , 2 )
		varianza.append( math.sqrt( float(aux/(rows)) ) )
	return varianza

# Metodo que calcula la funcion gaussiana de un dato.
def gaussian(x, avg, std):
	exponent = math.exp(-(math.pow(x-avg,2)/(2*math.pow(std,2))))
	return (1 / (math.sqrt(2*math.pi) * std)) * exponent

# Metodo que calcula el procentaje de exactitud total.
def total_accuracy(accu, test):
	correct = 0
	for x in range(len(test)):
		if test[x][-1] == accu[x][0]:
			correct += 1
	return (correct/float(len(test))) * 100.0

# Metodo que crea una tabla de hash de las clases
def hashtable(classes):
	mapa = {}
	i = 0
	for x in classes:
		mapa[x]=i
		i+=1
	return mapa

# Metodo que crea una matrix
def create_matrix(n_classes):
	m = [[0] * n_classes for i in range(n_classes)]
	return m

# Metodo que retorna una lista con las clases
def class_labels(classes):
	return list(classes.keys())

# Metodo que crea una matriz de confusion dado la prediccion y el test.
def confusion_matrix(accu, test):
	classes = class_labels(clases(test))
	mapa = hashtable(classes)
	m = create_matrix(len(classes))
	predicted = [tipo for tipo, prob in accu]
	expected = [test[i][-1] for i in range(len(test))]
	for pred, exp in zip(predicted, expected):
		m[mapa[pred]][mapa[exp]] += 1
	m = np.column_stack((classes,m))
	return m