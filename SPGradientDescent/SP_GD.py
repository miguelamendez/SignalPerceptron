import numpy as np
x,y= np.mgrid[-1:1:.1, -1:1:.1]

V = y*np.cos(np.pi*x) # just a random function for the potential

Ex,Ey= np.gradient(V)
print(Ex,Ey)

y_real=[0,1,1,0]
y_pred=[1,0,1,0]

def MSE(y_real,y_predict):
	#total= 1/((len(y_real))*np.sqrt(y_real-y_pred)
	total= np.sum(total)
	return total

def Signal_perceptron_gen(m,k):
	def signal_perceptron(theta,x):
		y_pred = 0
		for i in range(0,m**k):
			y_pred = ypred +  
	return signal_perceptron
	
def gradientDescent(x, y, theta, alpha, m,k, numIterations):
    xTrans = x.transpose()
    SP = Signal_perceptron_gen(m,k)
    for i in range(0, numIterations):
        print(x)
        hypothesis = SP(x, theta)
        hypothesis = np.exp(hypothesis*1j*np.pi/m)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta

def data_gen(m,k):
	xki=[]
	aix=np.zeros([k]);
	n=k  #No. of variables
	nn=m**n #|m^k| domain space
	nnn=m**nn #|Delta|=|m^m^k| function space
	j=0
	for xi in range(0,m**k,1):
		kx=xi;
		for xj in range(0,k,1): #Generamos los índices
			aix[xj]= int ( kx % m ); #Lo metemos en array 
			kx=int(kx/m); #siguientes índices
		print("aix=",aix)
		x=aix
		xki.append(x)
		j=j+1
	print(xki)



data_gen(2,2)	

