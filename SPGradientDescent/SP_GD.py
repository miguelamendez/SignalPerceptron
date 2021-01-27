import numpy as np
import random as rm

x,y= np.mgrid[-1:1:.1, -1:1:.1]

V = y*np.cos(np.pi*x) # just a random function for the potential

#Ex,Ey= np.gradient(V)
#print(Ex,Ey)

y_real=[0,1,1,0]
y_pred=[1,0,1,0]

def MSE(y_real,y_predict):
	#total= 1/((len(y_real))*np.sqrt(y_real-y_pred)
	total= np.sum(total)
	return total

def Signal_perceptron_gen(m,k):
	wki=[]
	aiw=np.zeros([k]);
	for i in range(0,m**k,1):
		kw=i;
		for j in range(0,k,1): #Generamos los índices
			aiw[j]= int ( kw % m ); #Lo metemos en array 
			kw=int(kw/m); #siguientes índices
		w=[]
		for l in aiw:
			w.append(l)
		wki.append(w)
	arrw = np.asarray(wki)
	print("frecuency matrix",arrw.shape)
	def signal_perceptron(theta,x):
		y_pred = 0
		#print("x",x.shape)
		x = np.transpose(x)
		#print("x trans",x.shape)
		exp=np.dot(arrw,x)
		#print("exponent",exp.shape)
		o_sp=np.exp(1j*np.pi*exp)
		print("after exponential",o_sp)
		print("theta vector",theta.shape)
		y_sp=np.dot(theta,o_sp)
		print("result",y_sp)
	return signal_perceptron
	
def gradientDescent(x, y, theta, alpha, m,k, numIterations):
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

def loss(y_label,y_pred):
	n=len(y_label)
	loss= (y_label-y_pred)**2
	loss= 1/n*(np.sum(loss))
	

def data_gen(m,k,y=0):
	alpha=np.ones([m**k])
#Creating dataset:
	xki=[]
	aix=np.zeros([k]);
	for i in range(0,m**k,1):
		kx=i;
		for j in range(0,k,1): #Generamos los índices
			aix[j]= int ( kx % m ); #Lo metemos en array 
			kx=int(kx/m); #siguientes índices
		x=[]
		for l in aix:
			x.append(l)
		xki.append(x)
	arrx = np.asarray(xki)
	#Boolean Funtion:
	if m==2 and y!=0:
		return arrx, y
	
	else:
	#Creating random function
		b=np.zeros([m**k])
		fun=rm.randint(0,m**m**k)
		r=fun
		for j in range(0,m**k):
			r=fun%m
			fun=int(fun/m)
			b[j]=r
		return arrx, b, alpha
		
x,y,a=data_gen(2,2)	
print(x,y,a)
signal=Signal_perceptron_gen(2,2)
signal(a,x)
