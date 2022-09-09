sigma= 5; %en la práctica no lo conozco
y= normrnd(0,sigma,1000,1);
m= jackknife(@var,y,1);
n= length(y);
bias= -sigma^2/n %known bias formula
jbias= (n-1)*(mean(m)-var(y,1))
