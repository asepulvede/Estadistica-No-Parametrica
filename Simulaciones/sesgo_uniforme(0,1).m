N=1000;
n=3000; 
X= rand(N,n);
X=X';
m= min(X);
hist(m)
E=mean(m) %valor esperado = 1/(n+1)
