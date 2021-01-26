import numpy as np

#Solving a sistem of linear equations
#print(X)
#A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
#B = np.array([25, -10, -4])
#X2 = np.linalg.solve(A,B)
#print(X2)

m=3;  #Módule (base) Ej: si m=3, se genera 0,1,2,0,1,2,...
k=2; #Number of nested loops. That is, number of variables
aix=np.zeros([k]); #Array of indexes (to order them)
aiw=np.zeros([k]); #Array of indexes (to order them)
ni=m**k   #Number of Iterations
n=k  #No. of variables
nn=m**n #|m^k| domain space
nnn=m**nn #|Delta|=|m^m^k| function space
# Matrix for  
A=np.zeros([nn,nn],dtype=complex)
#A=np.zeros([nn,nn]) 
divfrec=m-1
i=0; j=0
v=0; 
#Function for generating the matrix for m^k perceptron

for xi in range(0,ni,1):
	kx=xi;
	for xj in range(0,k,1): #Generamos los índices
		aix[xj]= int ( kx % m ); #Lo metemos en array 
		kx=int(kx/m); #siguientes índices
	#print("aix=",aix)
	j=0;
	#First Inner nested loop that generates all combinations of w for a signal
	for wi in range(0,ni,1):
		kw=wi;
		for wj in range(0,k,1): #Generamos los índices
			aiw[wj]= int ( kw % m ) ; #Lo metemos en array 
			kw=int(kw/m); #siguientes índices
			#print(i,j,A[i,j],"|",end='')
		exponente=0
		#Seconf Inner loop that  multiplies and sums
		for ii in range(0,k,1):
			exponente=exponente + aix[ii]*aiw[ii]
			exponente=int(exponente)
		#print("exponente=",exponente)
		exponente=1j*np.pi*exponente/divfrec
		#print(exponente)
		#print(np.exp(exponente))
		A[i,j]=np.exp(exponente)
		#print(A[i,j])
		j=j+1
		#print("aiw=",aiw,"j=",j)
		#for aj in range(0,nc,1):
		#	print(i,j,A[i,j],"|",end='')
		#	print()
	i=i+1
print("A=",A)
#a1=A.astype(int) only uncomment this if you dont want the matrix to use the complex part
atrans= A.transpose()
print(atrans)
#h=np.matmul(A,atrans)
#print(h)
#Function for calculating all posible functions from a m^k function space (Warning: this could fail if the function space is to big since it will require lots of memory)
def all_func():
	b=np.zeros([nnn,nn],dtype=complex)
	alfa=np.zeros([nnn,nn],dtype=complex)
	#Function that obtains the parameters of all possible functions
	for i in range(0,nnn):
		ii=i; r=i;
		#base transformation function
		for j in range(0,nn):
			r=ii%m
			ii=int(ii/m)
			b[i,j]=r
		#print(b[i])
		X = np.linalg.inv(A).dot(b[i])
		alfa[i]=X
		print("Function ",i,"):",b[i],alfa[i])
		#print(b[i])
		#print("weight:",alfa[i])
	#print(b)
	#print(alfa)
#Function for calculating all posible functions from a m^k function space (Warning: this could fail if the function space is to big since it will require lots of memory)
def single_func():
	#Some boolean functions of different arity: 2^2,2^3,2^4,2^5
	b5=np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0]) #arity 5
	b4=np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,0,0,0]) #arity 4
	b3=np.array([0,1,1,0,1,0,0,1]) #arity 3 XOR
	b2=np.array([0,1,1,0]) #arity 2 XOR
	#Some trilean functions of different arity: 3^2,3^3
	t3=np.array([0,1,2,1,1,2,0,2,1,0,1,1,0,1,0,0,1,1,1,1,2,0,2,1,0,1,1]) #arity 3 
	t2=np.array([0,1,2,1,1,2,0,2,1]) #arity 2 
	#Some m^k functions:
	
	p2=np.array([0,1,2,3,1,2,4,2,1,2,3,1,3,3,0,0,4,1,2,1,3,0,4,1,0]) 
	h3=np.array([5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,1,2,5,4,3,4,2,0,1,4,2,5,3,4,4,3]) 
	o2=([0,1,2,3,1,2,5,2,1,6,3,1,5,3,7,0,4,1,2,1,3,0,7,1,7,0,1,2,3,1,2,5,2,1,6,3,1,5,3,7,0,4,1,2,1,3,0,7,1,7,1,3,4,7,6,1,2,3,7,0,2,5,3,7])
	X = np.linalg.inv(A).dot(h3)
	print("parameters for function:",X)
all_func();
#single_func();
print("The function space size is: ",nnn)
print(" End of lpn")
