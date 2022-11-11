filename = 'Returns.txt';
[Return,~] = importdata(filename);
%%
Return(:,1) = [];
medias = mean(Return,2);
y = zeros(size(Return,1),1);
for i = 1:size(Return,1)
    if medias(i)>0
        y(i) = 1;
    else
        y(i) = -1;
    end
end

%% 

idx = find(medias<=0);
X = Return(idx,:);
%%
%Test de rangos
%Pasamos el test de MannWhitney
N = size(X,2); n = size(X,1);
MannWhitneyp = zeros(N,N); MannWhitneyh = zeros(N,N);
for i = 1:N
    Xi = X(:,i);
    for j = 1:N
        Xj = X(:,j);
        [p,h] = ranksum(Xi,Xj);
        MannWhitneyp(i,j) = p; MannWhitneyh(i,j) = h;
    end
end
I = MannWhitneyp <0.05
%Observamos que ninguno no pasa el test, entonces vamos a acortar la
%ventana de tiempo.
heatmap(double(I))
colormap spring
title('Test de rangos de Mann Whitney')
%% 
heatmap(MannWhitneyp)
colormap spring
title('Test de rangos de Mann Whitney')
%%
%%
%Para visualizar las curvas entre todos los activos y el área en común
%entre ellos


clf
at = zeros(N,N);
cont = 1;
for i = 1:N
    Xi = X(:,i);
    for j = 1:N
        Xj = X(:,j);
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



%Ahora veamos las densidades de los activos que no seon de la misma
%distribucion segun el test

%Si dejamos alpha = 0.05 tenemos los activos 1 y 6 (tenemos muchos pero
%escogemos segun el menor area en comun



Xi = X(:,1);
Xj = X(:,2);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
CM = summer(3)
   
plot(x1,f1,'Color',CM(1,:))
hold on
plot(x2,f2,'Color',CM(2,:))


title(are)
legend({'Activo 1','Activo 2'})
xlabel('Tiempo')
ylabel('PDF')

%%
newX = newData(X',10000);
newX = newX.Data;

%%
%Test de rangos
%Pasamos el test de MannWhitney

N = size(newX,2); n = size(newX,1);
MannWhitneyp = zeros(N,N); MannWhitneyh = zeros(N,N);
for i = 1:N
    Xi = newX(:,i);
    for j = 1:N
        Xj = newX(:,j);
        [p,h] = ranksum(Xi,Xj);
        MannWhitneyp(i,j) = p; MannWhitneyh(i,j) = h;
    end
end
%%
I = MannWhitneyp <0.05
%Observamos que ninguno no pasa el test, entonces vamos a acortar la
%ventana de tiempo.
heatmap(double(I))
colormap spring
title('Test de rangos de Mann Whitney')

%%
heatmap(MannWhitneyp)
colormap spring
title('Test de rangos de Mann Whitney')
%%
N = size(Return,2); n = size(Return,1);

X = Return;
for i = 1:N
    Min = min(Return); Max = max(Return);
    for j = 1:n
        X(j,i) = (X(j,i)- Min(i))/(Max(i) - Min(i)) + i/2;
    end
end
%%

%Graficar distribuciones empiricas
clf

figure(1)

CM = parula(17);
for j = 1:N
    
    [F,t] = ecdf(X(:,j));
    plot(t,F, 'color', CM(j,:))
    Area(j) = trapz(t,1-F);
    hold on
    xlabel('t')
    ylabel('ECDF')
    legend
    title('Distribuciones empiricas los retornos transformados')
end 

%%
[~, idx] = max(Area);
Best = Return(:,N);

qqplot(Best)
title('QQplot para el activo con mejores rendimientos')
%%
%Queremos 1, que p>0.05
[h,p] = jbtest(Best)
%%
%Queremos 1
[h,p]= kstest(Best)

%%
[mu, sigma] = normfit(Best)

%%
[f,xi] = ksdensity(Best);

y = normpdf(xi,mu,sigma);
figure(1)
plot(xi,f,'Color',CM(1,:))
hold on
plot(xi,y,'Color',CM(5,:))
legend({'Activo 12 modificado','Normal (\mu =0.832,\sigma = 6.519'})

xlabel('Tiempo')
ylabel('PDF')
title('Densidad aproximada vs estimada')

%%
[f,xi] = ecdf(Best);

y = normcdf(xi,mu,sigma);
figure(1)
plot(xi,f,'Color',CM(1,:))
hold on
plot(xi,y,'Color',CM(5,:))
legend({'Activo 12 modificado','Normal (\mu =0.832,\sigma = 6.519'})

xlabel('Tiempo')
ylabel('CDF')
title('Distribucion aproximada vs estimada')

%%
idx = find(medias<=0);
X = Return(idx,:);
idy = find(medias>0);
Y= Return(idy,:);
%% 


%Test de homogeneidad

[Ix,Iy,linmod] = ddplot2(X,Y)
title('Regresion lineal ddplot')
%%
[LM, PD1, PD2] = TestDepthTukey(X, Y);
%%
plot(LM)
title('Regresion lineal del test de profundidad de Tukey')

%%
X = Return(:,12); 

Y = sin(5.*X)+0.1*Return(:,10);

NadarayaWatson = ksr(X,Y);
CM = parula(17)
plot(X,Y,'o','Color',CM(12,:))
hold on
plot(NadarayaWatson.x,NadarayaWatson.f,'Color',CM(5,:))
xlabel('Activo 12')
ylabel('Sen(5*activo12)+0.1*activo10')
legend('Datos','Nadaraya Watson')
title('Regresion de Nadaraya Watson')

bandwidth  = NadarayaWatson.h;

Nada = interp1(NadarayaWatson.x,NadarayaWatson.f,X);

OrdinaryR2 = 1- sum((Y-Nada).^2)/sum((Y-mean(Y)).^2)
AjustedR2 =  OrdinaryR2*((length(Y)-1)/(length(Y)-2))
%%
boot = bootstrp(10000,@(x)(skewness(x) + 1)/(kurtosis(x) + 1),Return(:,12));
(skewness(Return(:,12)) + 1)/(kurtosis(Return(:,12)) + 1)
CIB = [prctile(boot,2.5) prctile(boot,97.5)] 
%% 
%Punto 2
X = Return(:,1:11);Y = Return(:,12);

OLS = fitlm(X,Y)
Pearson = RobustReg(X,Y,'Pearson')
Spearman = RobustReg(X,Y,'Spearman')
Kendall = RobustReg(X,Y,'Kendall')
fastMCD = RobustReg(X,Y,'fastMCD')
Shrinkage = RobustReg(X,Y,'Shrinkage')
%% 
X(:,10)=[];
OLS = fitlm(X,Y)
%% 
X(:,8)=[];
OLS = fitlm(X,Y)
%%
X(:,8)=[];
OLS = fitlm(X,Y)
%%
X(:,2)=[];
OLS = fitlm(X,Y)
%%
X(:,2)=[];
OLS = fitlm(X,Y)
%%
plot(OLS)
title('Regresion lineal con minimos cuadrados ordinarios')
xlabel('Activos 1 al 11')
ylabel('Activo 12')
%%

plot(Shrinkage.y, 'Color', CM(5,:),'HandleVisibility', 'off')
hold on
plot(Y, 'Color', CM(12,:),'HandleVisibility', 'off')
title('Regresion lineal robusta (shrinkage)')
xlabel('Tiempo')
ylabel('Rendimiento')


plot(NaN, 'DisplayName', 'Shrinkage','Color',CM(5,:));
plot(NaN, 'DisplayName', 'Activo 12','Color',CM(12,:));
legend

%%
plot(fitlm(Y,Shrinkage.y))
title('Regresion lineal robusta (shrinkage)')
xlabel('Activos 1 al 11')
ylabel('Activo 12')



%%
%Distancia de Mahalanobis
X = Return(:,1:11);Y = Return(:,12);
[f1,~] = size(X);
[f2,~] = size(Y);
data = [X Y];
n = size(X,1);
alpha = 0.2;
k = floor(n*alpha);
%% 

for j = 1:k
    Mahal= zeros(size(data,1));
    length(Mahal)
    for i = 1:size(data,1)
        Mahal(i) = mahal(data(i,:),data);
    end
    [~,idx] = max(Mahal);
    data(idx,:) = [];
end

%%
X = Return(:,1:11);Y = Return(:,12);
[f1,~] = size(X);
[f2,~] = size(Y);
data = [X Y];
n = size(X,1);
boo = true;
cont = 1;
while boo
    linmod = fitlm(data(:,1:11),data(:,12));
    R2(cont) = linmod.Rsquared.Adjusted;
    if R2(cont) > 0.9
        boo = false;
    end
    if cont == n
        boo = false;
    end
    Mahal= zeros(size(data,1));
    for i = 1:size(data,1)
        Mahal(i) = mahal(data(i,:),data);
    end
    [~,idx] = max(Mahal);
    data(idx,:) = [];
    cont = cont+1;
end


%%

figure(1)
CM = summer(17);
clf
plot(Return(:,1:11),Return(:,12),'.','Color',CM(5,:),'HandleVisibility', 'off')
hold on
plot(data(:,1:11),data(:,12),'.','Color',CM(15,:),'HandleVisibility', 'off')
title('Outliers de datos generados (Distancia de Mahalanobis)')
xlabel('X')
ylabel('Y')

plot(NaN, 'DisplayName', 'Outliers','Color',CM(5,:));
plot(NaN, 'DisplayName', 'Region central','Color',CM(15,:));
legend

%%
%Analisis residual del modelo de regresion lineal
X = Return(:,1:11);Y = Return(:,12);
linmod = fitlm(X,Y);
r2s(1) = linmod.Rsquared.Adjusted;
nooutliersX = X; nooutliersY = Y;

i=2;
bool= true;
while bool
    Residuals = linmod.Residuals.Raw;
    [~, idx] = max(abs(Residuals));
    nooutliersX(idx,:) = []; nooutliersY(idx) = [];
    linmod = fitlm(nooutliersX,nooutliersY);
    r2s(i) = linmod.Rsquared.Adjusted;

    if r2s(i) >=0.9
        bool= false;
    end
    i=i+1;
end
%% 

for i = 2: length(r2s)
    Residuals = linmod.Residuals.Raw;
    [~, idx] = max(abs(Residuals));
    nooutliersX(idx,:) = []; nooutliersY(idx) = [];
    linmod = fitlm(nooutliersX,nooutliersY);
    r2s(i) = linmod.Rsquared.Adjusted;
end

plot(r2s)
title('Ajuste del modelo a traves del metodo de eliminacion de residuales')
xlabel('Iteraciones')
ylabel('R2 ajustado')
%%
clf
plot(X,Y,'.','Color',CM(5,:),'HandleVisibility', 'off')
hold on
plot(nooutliersX,nooutliersY,'.','Color',CM(15,:),'HandleVisibility', 'off')
title('Outliers de activos que vienen de distinta distribucion (Residuales)')
xlabel('Activos 1 al 11')
ylabel('Activo 12')
plot(NaN, 'DisplayName', 'Outliers','Color',CM(5,:));
plot(NaN, 'DisplayName', 'Region central','Color',CM(15,:));
legend
legend()

%%
%eliminamos el activo menos significativo (activo 8)

nooutliersX(:,8) = [];
linmod = fitlm(nooutliersX,nooutliersY)

%%
%eliminamos el activo menos significativo (activo 10)

nooutliersX(:,9) = [];
linmod = fitlm(nooutliersX,nooutliersY)
clf
plot(X,Y,'.','Color',CM(5,:),'HandleVisibility', 'off')
hold on
plot(nooutliersX,nooutliersY,'.','Color',CM(15,:),'HandleVisibility', 'off')
title('Outliers de activos que vienen de distinta distribucion (Residuales)')
xlabel('Activos 1 al 11 (menos 8 y 10)')
ylabel('Activo 12')
plot(NaN, 'DisplayName', 'Outliers','Color',CM(5,:));
plot(NaN, 'DisplayName', 'Region central','Color',CM(15,:));
legend

%%
X = Return(:,1:11);Y = Return(:,12);
Reg = RobustReg(X,Y,'Rho');

plot(fitlm(Reg.y,Y))
title('Regresion robusta vs Y')
%%
%inferencia bootstrap para los coeficientes
m = size(X,1); n = 100
for i = 1:n
    ind = randi([1 m],m,1);
    Xi = X(ind,:); Yi = Y(ind);
    Reg = RobustReg(X,Y,'Rho');
    betas = Reg.Betas;
    b1(i) = betas(1);
end

cinterval_bootstrap= [prctile(b1,2.5) prctile(b1,97.5)] 
%%
n = 1000;
X = exprnd(1,n,1);

jack = jackknife(@min,X);
jbias = (n-1)* (mean(jack)-min(X))