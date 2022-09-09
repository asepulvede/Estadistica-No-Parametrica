sigma= 5; %en la práctica no lo conozco
y= normrnd(0,sigma,100,1);
m= jackknife(@mean,y,1);
n= length(y);
bias= 0; %known bias formula
jbias= (n-1)*(mean(m)-mean(y,1))
