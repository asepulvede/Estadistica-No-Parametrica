y= 10+2*randn(200,1); %genero datos normales con media 10, desviación 2
m= bootstrp(100,@mean,y);
mean(m) % E(xbar) = mu
std(m) % sigma/sqrt(n)
hist(m)
%intervalo de confianza paramétrico= [xbar-ts&/sqrt(n) xbar+ts&/sqrt(n)]
[h p ci] = ttest(y) %h->hipotesis, p->p_value, ci-> intervalo de confianaza paramétrico
cinterval_bootstrap= [prctile(m,2.5) prctile(m,97.5)] 
