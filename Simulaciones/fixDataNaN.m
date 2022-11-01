% paso 1: tenemos una muestrada multivariante
filename = 'return.txt';
[X,~]=importdata(filename);

X_contamin= X;
n_tocontamin= randperm(size(X,1), round(0.05*size(X,1)));
for i=1:length(n_tocontamin)
    var_contam= randi([1 size(X,2)],1);
    X_contamin(i,var_contam)= nan; 
    
end
[rows, columns] = find(isnan(X_contamin));

[rows_sort, sortIdx]= sort(rows);
colums_sort= columns(sortIdx);

org=[];
for i=1:length(rows_sort)
    org(i)= X(rows_sort(i),colums_sort(i));
end

data_nan= X_contamin(rows_sort, :);
data_orig_nan= X(rows_sort,:);
data_no_nan= X_contamin(sum(isnan(X_contamin),2)==0,:);
nuevos_datos= multivariateDataGenerator(data_no_nan,3000);

%% 

dataxy= [data_no_nan; nuevos_datos];
dimen= size(data_nan);
imputados=[];
for i=1:dimen(1)
    dato= data_nan(i,:);
    [data_nan(i,:), imp]= metrica(dato,dataxy,2);
    imputados(i)= imp;
end

reg= fitlm(org,imputados)
clf
plot(reg)
