x = readtable('return.txt');
x = table2array(x);

%1.a)

%%Primero tomamos los últimos 900 meses

x = x(end-899:end,:);
N = size(x,2); n = size(x,1);


%Pasamos el test de MannWhitney
MannWhitneyp = zeros(N,N); MannWhitneyh = zeros(N,N);
for i = 1:N
    Xi = x(:,i);
    for j = 1:N
        Xj = x(:,j);
        [p,h] = ranksum(Xi,Xj);
        MannWhitneyp(i,j) = p; MannWhitneyh(i,j) = h;
    end
end
I = MannWhitneyp <0.1
%Observamos que ninguno no pasa el test, entonces vamos a acortar la
%ventana de tiempo.
%%
figure(1)
heatmap(MannWhitneyp, 'ColorLimits',[0 1])
colormap parula
title('p-valores para el test de rangos de Mann Whitney')

figure(2)
heatmap(double(I),'ColorLimits',[0 1])
colormap parula
title('Activos que pasan el test de Mann Whitney (alpha = 0.1)')

%%
x = x(1:450,:);
N = size(x,2); n = size(x,1);


%Pasamos el test de MannWhitney
MannWhitneyp = zeros(N,N); MannWhitneyh = zeros(N,N);
for i = 1:N
    Xi = x(:,i);
    for j = 1:N
        Xj = x(:,j);
        [p,h] = ranksum(Xi,Xj);
        MannWhitneyp(i,j) = p; MannWhitneyh(i,j) = h;
    end
end

I = MannWhitneyp <0.1
%%
figure(1)
heatmap(MannWhitneyp)
colormap parula
title('p-valores para el test de rangos de Mann Whitney')

figure(2)
heatmap(double(I))
colormap parula
title('Activos que pasan el test de Mann Whitney (alpha = 0.1)')
%%
%Para visualizar las curvas entre todos los activos y el área en común
%entre ellos
clf
at = zeros(N,N);
cont = 1;
for i = 1:N
    Xi = x(:,i);
    for j = 1:N
        Xj = x(:,j);
        mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
        pts = (mini:(maxi-mini)/100:maxi);
        [f1,x1] = ksdensity(Xi,pts); 
        [f2,x2] = ksdensity(Xj,pts); 
        dif = abs(f1-f2);
        are = abs(1-trapz(pts,dif));
        at(i,j) = are;
        subplot(N,N,cont)

        
        plot(x1,f1,'b')
        hold on
        plot(x2,f2,'r')
        title(are)
        cont = cont+1;
    end
end

%%

cont = 1;
for i = 1:N
    Xi = x(:,i);
    li = min(Xi); ls = max(Xi);
    xim = li:0.05:ls;
    Xipdf = fitdist(Xi,'Kernel');
    Xipdf = pdf(Xipdf,xim);
    for j = 1:N
        Xj = x(:,j);
        
        li = min(Xj); ls = max(Xj);
        xjm = li:0.05:ls;
        Xjpdf = fitdist(Xj,'Kernel');
        Xjpdf = pdf(Xjpdf,xjm);

        subplot(N,N,cont)
        plot(xim,Xipdf,'Color','b','LineWidth',2)
        hold on
        plot(xjm,Xjpdf,'Color','g','LineWidth',2)
        cont = cont+1;
    end
end

%%
%Ahora veamos las densidades de los activos que no seon de la misma
%distribucion segun el test

%Si dejamos alpha = 0.05 solo tenemos los activos 5 y 6.
%Con alpha = 0.1 tenemos los activos (4,6),(5,6).


cont = 1;
Xi = x(:,4);
Xj = x(:,6);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(2,1,1)
   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)
legend({'Activo 4','Activo 6'})

cont = cont+1;




Xi = x(:,5);
Xj = x(:,6);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(2,1,2)
   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)
legend({'Activo 5','Activo 6'})

sgtitle('Densidad activos con diferente distribucion segun Mann Whitney Test')

%%
%Ahora veamos las densidades de los activos que no seon de la misma
%distribucion segun el test

%Como la mayoría lo pasan, escogemos los que tengan la mayor área en común.
%(2,5), (7,9), (3,10)

cont = 1;
Xi = x(:,2);
Xj = x(:,5);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(3,1,1)
   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)
legend({'Activo 2','Activo 5'})

cont = cont+1;

Xi = x(:,7);
Xj = x(:,9);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(3,1,2)
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)
legend({'Activo 7','Activo 9'})

cont = cont+1;

Xi = x(:,3);
Xj = x(:,10);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(3,1,3)
   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)

legend({'Activo 3','Activo 10'})

sgtitle('Densidad activos con misma distribucion segun Mann Whitney Test')

%%
%1.b Coger los dos que vengan de distinta distribución
%vamos a tomar los 1100 datos otra vez
%x será el activo 5 y y el activo 6.

x = readtable('return.txt');
x = table2array(x);

X = x(:,5); Y = x(:,6);

%1.b.1. modelo de regresion

linmod = fitlm(X,Y);
plot(linmod)
title('Regresion lineal entre activos de diferente distribucion')
xlabel('Activo 5')
ylabel('Activo 6')

Residuals = linmod.Residuals.Raw;
%%
%inferencia bootstrap para los coeficientes
m = size(X,1); n = 100
for i = 1:n
    ind = randi([1 m],m,1);
    Xi = X(ind); Yi = Y(ind);
    linmod = fitlm(Xi,Yi);
    betas = linmod.Coefficients.Estimate;
    b1(i) = betas(2);b0(i) = betas(1);
end
%%
figure(1)
hist(b1)
title('Histograma de la inferencia bootstrap para \beta_1')
figure(2)
hist(b0)
title('Histograma de la inferencia bootstrap para \beta_0')

%%

%1.b.2

x = readtable('return.txt');
x = table2array(x);

X = x(:,5); Y = x(:,6);


%[LM, PD1, PD2] = TestDepthTukey(X,Y)
%%
[f1,~] = size(X);
[f2,~] = size(Y);
data = [X Y];

n = 100;%direcciones aleatorias

for i = 1:f1
    Tukey(i) = depthTukey(data(i,:),data,n);
end

alpha = 0.25;

depthMin = prctile(Tukey,alpha*100);
idx = Tukey >= depthMin;
XTukey = X(idx); YTukey = Y(idx);
plot(X,Y,'.r')
hold on
plot(X(idx),Y(idx),'.b')
title('Outliers de activos que vienen de distinta distribucion (Profundidad de Tukey)')
xlabel('Activo 5')
ylabel('Activo 6')
legend({'Outliers','Region central'})
%%
%Distancia de Mahalanobis
[f1,~] = size(X);
[f2,~] = size(Y);
data = [X Y];
for i = 1:f1
    Mahal(i) = mahal(data(i,:),data);
end

alpha = 0.75;

depthMin = prctile(Mahal,alpha*100);
idx = Mahal <= depthMin;
XMahal = X(idx); YMahal = Y(idx);
plot(X,Y,'.r')
hold on
plot(X(idx),Y(idx),'.b')
title('Outliers de activos que vienen de distinta distribucion (Distancia de Mahalanobis)')
xlabel('Activo 5')
ylabel('Activo 6')
legend({'Outliers','Region central'})
%%
%Regresion sobre la region central Tukey
linmod = fitlm(XTukey,YTukey)
plot(linmod)
title('Regresion lineal sobre los datos de la region central (profundidad Tukey)')
xlabel('Activo 5')
ylabel('Activo 6')
%%
%Regresion sobre la region central Mahal
linmod = fitlm(XMahal,YMahal)
plot(linmod)
title('Regresion lineal sobre los datos de la region central (distancia de Mahalanobis)')
xlabel('Activo 5')
ylabel('Activo 6')

%%
%1.b.3 Analisis residual del modelo de regresion lineal
linmod = fitlm(X,Y);
r2s = zeros(floor(size(X,1)*0.25)+1,1);
r2s(1) = linmod.Rsquared.Adjusted;
nooutliersX = X; nooutliersY = Y;

for i = 2: length(r2s)
    Residuals = linmod.Residuals.Raw;
    [~, idx] = max(abs(Residuals));
    nooutliersX(idx) = []; nooutliersY(idx) = [];
    linmod = fitlm(nooutliersX,nooutliersY);
    r2s(i) = linmod.Rsquared.Adjusted;
end

plot(r2s)
title('Ajuste del modelo a traves del metodo de eliminacion de residuales')
xlabel('Iteraciones')
ylabel('R2 ajustado')
%%

plot(linmod)
title('Regresion lineal de la ultima iteracion')
xlabel('Activo 5')
ylabel('Activo 6')

%%
plot(X,Y,'.r')
hold on
plot(nooutliersX,nooutliersY,'.b')
title('Outliers de activos que vienen de distinta distribucion (Residuales)')
xlabel('Activo 5')
ylabel('Activo 6')
legend({'Outliers','Region central'})


%%
%1.b.4 Modelos de regresion robustos y no parametricos

%fastMCD
covs = robustcov([X,Y]);
covxx= covs(1,1); covxy= covs(1,2);
b1= covxy/covxx;
b0= mean(Y)-b1*mean(X);

reg = b0 + b1*X;

OrdinaryR2 = 1- sum((Y-reg).^2)/sum((Y-mean(Y)).^2);
AjustedR2 =  OrdinaryR2*((length(Y)-1)/(length(Y)-2));

linmod = fitlm(X,Y);
coeffs = linmod.Coefficients.Estimate;

plot(X,Y,'.')
hold on
plot(X,reg,'--')
hold on
plot(X, coeffs(1)+coeffs(2)*X )

legend('Robust','data','Reg normal')
title('Regresion Robusta y No Parametrica (MCD) vs Regresion Parametrica')
xlabel('Activo 5')
ylabel('Activo 6')
hold off
regMCD = reg;
AjustedR2MCD = AjustedR2;
%%
%inferencia bootstrap para los coeficientes
m = size(X,1); n = 100
beta1 = zeros(n,1);beta0 = zeros(n,1);
for i = 1:n
    ind = randi([1 m],m,1);
    Xi = X(ind); Yi = Y(ind);
    covs = robustcov([Xi,Yi]);
    covxx = covs(1,1); covxy = covs(1,2);
    b1 = covxy/covxx;
    b0 = mean(Yi)-b1*mean(Xi);
    beta1(i) = b1;beta0(i) = b0;
end
%% 

lo = prctile(beta1,2.5); hi = prctile(beta1,97.5); 
intBeta1 = [lo hi];
lo = prctile(beta0,2.5); hi = prctile(beta0,97.5); 
intBeta0 = [lo hi];

%%
%Kendall
covs = robustcov([X,Y]);
covxx = corr(X,X,'type','Kendall')*std(X)*std(X); covxy = corr(X,Y,'type','Kendall')*std(X)*std(Y);
b1 = covxy/covxx;
b0 = mean(Y)-b1*mean(X);

reg = b0 + b1*X;

OrdinaryR2 = 1- sum((Y-reg).^2)/sum((Y-mean(Y)).^2);
AjustedR2 =  OrdinaryR2*((length(Y)-1)/(length(Y)-2));

plot(X,reg,'g')
hold on
plot(fitlm(X,Y))
legend('Robust','data','Reg normal')
title('Regresion Robusta y No Parametrica (Kendall) vs Regresion Parametrica')
xlabel('Activo 5')
ylabel('Activo 6')
hold off

regKendall = reg;
AjustedR2Kendall = AjustedR2;
%%
%inferencia bootstrap para los coeficientes
m = size(X,1); n = 100
beta1 = zeros(n,1);beta0 = zeros(n,1);

for i = 1:n
    ind = randi([1 m],m,1);
    Xi = X(ind); Yi = Y(ind);
    covs = robustcov([Xi,Yi]);
    covxx = corr(Xi,Xi,'type','Kendall')*std(Xi)*std(Xi); covxy = corr(Xi,Yi,'type','Kendall')*std(Xi)*std(Yi);
    b1 = covxy/covxx;
    b0 = mean(Yi)-b1*mean(Xi);
    beta1(i) = b1;beta0(i) = b0;
end

lo = prctile(beta1,2.5); hi = prctile(beta1,97.5); 
intBeta1 = [lo hi];
lo = prctile(beta0,2.5); hi = prctile(beta0,97.5); 
intBeta0 = [lo hi];

%%
%Spearman

covxx = corr(X,X,'type','Spearman')*std(X)*std(X); covxy = corr(X,Y,'type','Spearman')*std(X)*std(Y);
b1 = covxy/covxx;
b0 = mean(Y)-b1*mean(X);

reg = b0 + b1*X;

OrdinaryR2 = 1- sum((Y-reg).^2)/sum((Y-mean(Y)).^2);
AjustedR2 =  OrdinaryR2*((length(Y)-1)/(length(Y)-2));

plot(X,reg,'--k')
hold on
plot(fitlm(X,Y))
legend('Robust','data','Reg normal')
title('Regresion Robusta y No Parametrica (Spearman) vs Regresion Parametrica')
xlabel('Activo 5')
ylabel('Activo 6')
hold off

regSpearman = reg;
AjustedR2Spearman= AjustedR2;
%%
%inferencia bootstrap para los coeficientes
m = size(X,1); n = 100
beta1 = zeros(n,1);beta0 = zeros(n,1);

for i = 1:n
    ind = randi([1 m],m,1);
    Xi = X(ind); Yi = Y(ind);
    covxx = corr(Xi,Xi,'type','Spearman')*std(Xi)*std(Xi); covxy = corr(Xi,Yi,'type','Spearman')*std(Xi)*std(Yi);
    b1 = covxy/covxx;
    b0 = mean(Yi)-b1*mean(Xi);
    beta1(i) = b1;beta0(i) = b0;
end

lo = prctile(beta1,2.5); hi = prctile(beta1,97.5); 
intBeta1 = [lo hi];
lo = prctile(beta0,2.5); hi = prctile(beta0,97.5); 
intBeta0 = [lo hi];
%%
%Combinations
covs = robustcov([X,Y]);
covxx = covs(1,1); covxy = corr(X,Y,'type','Spearman')*std(X)*std(Y);
b1 = covxy/covxx;
b0 = mean(Y)-b1*mean(X);

reg = b0 + b1*X;

OrdinaryR2 = 1- sum((Y-reg).^2)/sum((Y-mean(Y)).^2);
AjustedR2 =  OrdinaryR2*((length(Y)-1)/(length(Y)-2));

plot(X,reg,'--k')
hold on
plot(fitlm(X,Y))
legend('Robust','data','Reg normal')
title('Regresion Robusta y No Parametrica (MCD y Spearman) vs Regresion Parametrica')
xlabel('Activo 5')
ylabel('Activo 6')
hold off

regMCDSpearman = reg;
AjustedR2MCDSpearman = AjustedR2;
%%
%inferencia bootstrap para los coeficientes
m = size(X,1); n = 100
beta1 = zeros(n,1);beta0 = zeros(n,1);

for i = 1:n
    ind = randi([1 m],m,1);
    Xi = X(ind); Yi = Y(ind);
    covs = robustcov([Xi,Yi]);
    covxx = covs(1,1); covxy = corr(Xi,Yi,'type','Spearman')*std(Xi)*std(Yi);
    b1 = covxy/covxx;
    b0 = mean(Yi)-b1*mean(Xi);
    beta1(i) = b1;beta0(i) = b0;
end

lo = prctile(beta1,2.5); hi = prctile(beta1,97.5); 
intBeta1 = [lo hi];
lo = prctile(beta0,2.5); hi = prctile(beta0,97.5); 
intBeta0 = [lo hi];
%%
%Shrinkage Rao-Blackwell estimator
covs = shrinkage_cov([X,Y],'rblw');
covxx = covs(1,1); covxy = covs(1,2);
b1 = covxy/covxx;
b0 = mean(Y)-b1*mean(X);

reg = b0 + b1*X;

OrdinaryR2 = 1- sum((Y-reg).^2)/sum((Y-mean(Y)).^2);
AjustedR2 =  OrdinaryR2*((length(Y)-1)/(length(Y)-2));

plot(X,reg,'--k')
hold on
plot(fitlm(X,Y))
legend('Robust','data','Reg normal')
title('Regresion Robusta y No Parametrica (Shrinkage Rao-Blackwell) vs Regresion Parametrica')
xlabel('Activo 5')
ylabel('Activo 6')
hold off
regShrinkage = reg;
AjustedR2Shrinkage = AjustedR2;
%%
%inferencia bootstrap para los coeficientes
m = size(X,1); n = 100
beta1 = zeros(n,1);beta0 = zeros(n,1);

for i = 1:n
    ind = randi([1 m],m,1);
    Xi = X(ind); Yi = Y(ind);
    covs = shrinkage_cov([Xi,Yi],'rblw');
    covxx = covs(1,1); covxy = covs(1,2);
    b1 = covxy/covxx;
    b0 = mean(Yi)-b1*mean(Xi);
    beta1(i) = b1;beta0(i) = b0;
end

lo = prctile(beta1,2.5); hi = prctile(beta1,97.5); 
intBeta1 = [lo hi];
lo = prctile(beta0,2.5); hi = prctile(beta0,97.5); 
intBeta0 = [lo hi];
%%
%plot all together
linmod = fitlm(X,Y);
coeffs = linmod.Coefficients.Estimate;

plot(X,Y,'.','Color','#f1ee8e')
hold on
plot(X,reg,'--','LineWidth',1.2)

plot(X, coeffs(1)+coeffs(2)*X,'LineWidth',1.2 )
plot(X,regMCD,'--','LineWidth',1.2)

plot(X,regKendall,'--','LineWidth',1.2)

plot(X,regSpearman,'--','LineWidth',1.2)

plot(X,regMCDSpearman,'--','LineWidth',1.2)

plot(X,regShrinkage,'--','LineWidth',1.2)

%plot(fitlm(X,Y))
legend('Datos','Regresion fitlm','Fast MCD','Kendall','Spearman','MCD and Spearman','Shrinkage')
title('Regresion Robusta y No Parametrica vs Regresion Parametrica')
xlabel('Activo 5')
ylabel('Activo 6')
hold off
regShrinkage = reg;
AjustedR2Shrinkage = AjustedR2;

%% 
%1.c
na =10000; nb = 15000-na;
xa = 6+2*rand(na,1); xb = 2+8*rand(nb,1);
aux_a = randperm(size(xa,1), round(0.2*size(xa,1)));
aux_b = randperm(size(xb,1), round(0.8*size(xb,1)));

xa_a = xa(aux_a); xb_b = xb(aux_b);

ya = normrnd(-2*xa_a+10,1);
yb = normrnd(2*xb_b+4,1);

X = cat(1,xa_a,xb_b); Y = cat(1,ya,yb);

mini = min(Y); maxi = max(Y);
pts = (mini:(maxi-mini)/100:maxi);
[fy,xy] = ksdensity(Y,pts); 
plot(xy,fy)
%% 

regr= fitlm(X,Y);
plot(regr)
xlabel('X')
ylabel('Y')
title('Regresion lineal entre X y Y')
legend({'Datos','Ajuste','Bandas de confianza'})


%%
Xi = X;
Xj = Y;
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    

   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title('Densidades de X y de Y',are)

legend({'Y','X'})

%%
%Volver a hacer el punto 1.b para estos datos


%%
%1.b.2
[f1,~] = size(X);
[f2,~] = size(Y);
data = [X Y];

n = 100;%direcciones aleatorias

for i = 1:f1
    Tukey(i) = depthTukey(data(i,:),data,n);
end

alpha = 0.25;

depthMin = prctile(Tukey,alpha*100);
idx = Tukey >= depthMin;
XTukey = X(idx); YTukey = Y(idx);
plot(X,Y,'.r')
hold on
plot(X(idx),Y(idx),'.b')
title('Outliers de datos generados (Profundidad de Tukey)')
xlabel('X')
ylabel('Y')
legend({'Outliers','Region central'})

%%
%Distancia de Mahalanobis
[f1,~] = size(X);
[f2,~] = size(Y);
data = [X Y];
for i = 1:f1
    Mahal(i) = mahal(data(i,:),data);
end

alpha = 0.75;

depthMin = prctile(Mahal,alpha*100);
idx = Mahal <= depthMin;

XMahal = X(idx); YMahal = Y(idx);
plot(X,Y,'.r')
hold on
plot(X(idx),Y(idx),'.b')
title('Outliers de datos generados (Distancia de Mahalanobis)')
xlabel('X')
ylabel('Y')
legend({'Outliers','Region central'})
%%
%Regresion sobre la region central Tukey
linmod = fitlm(XTukey,YTukey)
plot(linmod)
title('Regresion lineal sobre los datos de la region central (profundidad Tukey)')
xlabel('X')
ylabel('Y')
%%
%Regresion sobre la region central Mahal
linmod = fitlm(XMahal,YMahal)
plot(linmod)
title('Regresion lineal sobre los datos de la region central (distancia de Mahalanobis)')
xlabel('X')
ylabel('Y')

%%
%1.b.3 Analisis residual del modelo de regresion lineal
linmod = fitlm(X,Y);
r2s = zeros(floor(size(X,1)*0.25)+1,1);
r2s(1) = linmod.Rsquared.Adjusted;
nooutliersX = X; nooutliersY = Y;
for i = 2: length(r2s)
    Residuals = linmod.Residuals.Raw;
    [~, idx] = max(abs(Residuals));
    nooutliersX(idx) = []; nooutliersY(idx) = [];
    linmod = fitlm(nooutliersX,nooutliersY);
    r2s(i) = linmod.Rsquared.Adjusted;

end

plot(r2s)
title('Ajuste del modelo a traves del metodo de eliminacion de residuales')
xlabel('Iteraciones')
ylabel('R2 ajustado')

%%
plot(X,Y,'.r')
hold on
plot(nooutliersX,nooutliersY,'.b')
title('Outliers de datos generados (Residuales)')
xlabel('X')
ylabel('Y')
legend({'Outliers','Region central'})

%%
%1.d


%Del mapa de correlaciones escogemos los mas correlacionados(3,10).
x = readtable('return.txt');
x = table2array(x);


X = x(:,3); Y = x(:,10);

ksr(X,Y)
title('Regresion de Nadaraya Watson')
xlabel('Activo 3 (X)')
ylabel('Activo 10 (Y)')
legend({'Ajuste','Datos'},'Location', 'Best')
%%
ksr(X,Y.^2)
title('Regresion de Nadaraya Watson')
xlabel('Activo 3 (X)')
ylabel('Activo 10 (Y^2)')
legend({'Ajuste','Datos'},'Location', 'Best')
%%
ksr(X,Y.^5)
title('Regresion de Nadaraya Watson')
xlabel('Activo 3 (X)')
ylabel('Activo 10 (Y^5)')
legend({'Ajuste','Datos'},'Location', 'Best')
%%
ksr(X,Y.^(1/5))
title('Regresion de Nadaraya Watson')
xlabel('Activo 3 (X)')
ylabel('Activo 10 (Y^{1/5})')
legend({'Ajuste','Datos'},'Location', 'Best')
%%
ksr(X,cos(Y))
title('Regresion de Nadaraya Watson')
xlabel('Activo 3 (X)')
ylabel('Activo 10 (cos(Y))')
legend({'Ajuste','Datos'},'Location', 'Best')
%%
%1.e.

vida = readtable('vida.txt');
vida = table2array(vida);
vida = vida(:,1:5);
idx = randi([1 size(vida,1)],1);
vidaX = vida(1:idx,:);vidaY = vida(idx+1:end,:);
%%
[Ix,Iy,mdl] = ddplot2(vidaX,vidaY)
title('Vida(1:272) vs Vida(273:500)')
xlabel('Vida(1:272)')
ylabel('Vida(272:500)')
%%
%1.f.

returns = readtable('return.txt');
returns = table2array(returns);

idx = randi([1 size(returns,1)],1);
returnsX = returns(1:idx,:);returnsY = returns(idx+1:end,:);
%%
[Ix,Iy,mdl] = ddplot2(returnsX,returnsY)
title('Return(1:536) vs Return(537:1067)')
xlabel('Return(1:536)')
ylabel('Return(537:1067)')




