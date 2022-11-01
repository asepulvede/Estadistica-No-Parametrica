function [lista_sin_nan,imputado] = metrica(lista_nan,dataxy,norma)
%Métrica que defina con qué dato se va a llenar el nan
% lista_nan = lista contamidad, dataxy= el dataset de donde provino
% norma= norma de la métrica: 1,2,inf
lista1= lista_nan;
index= find(isnan(lista_nan));
lista1(index)=[];
datasin_var= dataxy;
datasin_var(:,index)=[];
dimen= size(datasin_var);
if norma==1
d= zeros(dimen(1),length(lista1));
for i=1:dimen(1)
    for j=1:length(lista1)
        d(i,j)= norm(lista1(j)-datasin_var(i,j));
    end
    sumfilas= sum(d,2);
end
indice_min= find(sumfilas== min(sumfilas));
lista_nan(index)= dataxy(indice_min(1),index);
imputado=dataxy(indice_min(1),index);
lista_sin_nan = lista_nan;

elseif norma==2
    d= zeros(dimen(1));
    for i=1:length(d)
        d(i)= norm(lista1-datasin_var(i,:));
    end
    indice_min= find(d==min(d));
    lista_nan(index)= dataxy(indice_min(1),index);
    imputado=dataxy(indice_min(1),index);
    lista_sin_nan = lista_nan;
else
    d= zeros(dimen(1));
    for i=1:length(d)
        d(i)= norm(lista1-datasin_var(i,:),inf);
    end
    indice_min= find(d==min(d));
    lista_nan(index)= dataxy(indice_min(1),index);
    imputado=dataxy(indice_min(1),index);
    lista_sin_nan = lista_nan;
end

end
