n = 5000;
xi = exprnd(1,n,1);
[F,t] = ecdf(xi);
clf
plot(t,F)
A = trapz(t,1-F)
xbar = sum(xi)/n
% Añadir ruido
n = 500;
ruido = 200;
xi = exprnd(1,n,1);
xj = [xi;ruido];
[F,t] = ecdf(xi);
clf
plot(t,F)
A = trapz(t,1-F)
xbar = sum(xj)/n
