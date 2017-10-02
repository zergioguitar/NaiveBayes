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
from sfloresLib  import *
from naive_bayes import *
import numpy as np
import argparse 

#######  Functions  #######
def GUI( ):
	ventana = createWindow()
	ventana.setTitle("Naive Bayes")
	ventana.setSize(600,600) 
	ventana.setColor("white")
	ventana.setCloseConfirm()
	ventana.createButton("Browse" , browse)
	return ventana

def browse( ):
	file_name = ventana.askopenfile((("CSV", "*.csv"),("All files", "*.*")))
	test(file_name, True)

def read_file(filename, delimiter):
	aux = open(filename).read()
	aux = [item.split(delimiter) for item in aux.split('\n')[:]]
	for i in range(len(aux)):
		for j in range(len(aux[i])-1):
			aux[i][j] = float(aux[i][j])
	#desordenamos los datos
	indexes = np.random.permutation(len(aux))
	aux = np.asarray(aux)[indexes].tolist()
	return aux

def test(database, plot):
	#leemos la base de datos
	iris = read_file(database, ",")
	train = iris[ : int(len(iris)*0.9)] #seleccionamos el 90% de los datos
	test = iris[int(len(iris)*0.9) : ] #seleccionamos el 10% de los datos

	modelo = NaiveBayes()
	result = modelo.train_model(train)
	pred = modelo.probability(test)
	accu = modelo.predict(pred)
	matriz = confusion_matrix(accu,test)
 
	print("Confusion Matrix: \n", matriz )
	print("Model accuracy: ", total_accuracy(accu,test))

	#graficamos el modelo, usando el primer atributo.
	modelo.graph()
	

#######  Main Program  #######
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', metavar='filename', action="store", dest="file", help='database file path name.')
    args = parser.parse_args()
    if(args.file == None):
    	ventana = GUI()
    	ventana.keepAlive()
    else:
    	test(args.file, False)
