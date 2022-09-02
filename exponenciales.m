N=500;
n=100;
X= -log(rand(N,n));
D1= X(:,1);
figure(1)
hist(D1)
%%
X=X'
Dmin= min(X); %minimo por fila (de cada muestra)
figure(2)
hist(Dmin)
%%
Dmax= max(X);
figure(3)
hist(Dmax)
