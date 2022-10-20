% paso 1: tenemos una muestrada multivariante
filename = 'return.txt';
[X,delimiterOut]=importdata(filename);
dimensions= size(X);
n=dimensions(1); k=dimensions(2);
mins= floor(min(X));
maxs= ceil(max(X));
numero_clases= floor(sqrt(length(X)));
rangos= maxs-mins;
rangos_= rangos/numero_clases;

tabla_f= zeros(numero_clases,k);

m_clases= zeros(k,numero_clases+1);
for i=1:k
    clases= mins(i):rangos_(i):maxs(i);
    m_clases(i,:) = clases;  
end

for i=1:k
    for j=1:numero_clases
        ls=find(X(:,i)>=m_clases(i,j) & X(:,i)<m_clases(i,j+1));
        tabla_f(j,i)= length(ls);
    end
end

frec_var= (1/length(X))*tabla_f;
frec_abs= cumsum(frec_var);
frec_abs(end,:)=1;

empiricas= zeros(n,k);
for i=1:k
    for j=1:n
        aux= find(X(:,i)<= X(j,i));
        empiricas(j,i)= length(aux)/length(X);
    end
end

n_datos= 5000;
nuevos_datos = zeros(n_datos,k);

for w=1:n_datos
    uniforme= randi([1 n], 1,1);
    val_em= empiricas(uniforme,:);
    nuevo_dato = zeros(k,1);
    for i= 1:length(val_em)
        aux2= find(frec_abs(:,i) >=val_em(i));
        nuevo_dato(i) =  (m_clases(i,aux2(1)+1) - m_clases(i,aux2(1)))*rand(1) + m_clases(i,aux2(1));
    end
    nuevos_datos(w,:)= nuevo_dato;
end

for i =1:k
    xi = X(:,i);
    ai = nuevos_datos(:,i);
    subplot(k,2,(2*i-1))
    histogram(xi)
    subplot(k,2,(2*i))
    histogram(ai)
end

for i =1:k

    xi = X(:,i);

    ai = nuevos_datos(:,i);
    [f1,x1] = ksdensity(xi); 
    [f2,x2] = ksdensity(ai); 
    dif = abs(f1-f2);

    %a = 1-trapz(pts,dif)
    subplot(k,1,i)
    plot(x1,f1)
    hold on
    plot(x2,f2)
end
