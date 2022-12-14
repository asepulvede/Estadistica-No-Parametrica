function Cov = RobustCov(X,Y, method)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

N = size(X,2); n = size(X,1);

if strcmp(method, 'Kendall') | strcmp(method, 'Spearman')| strcmp(method, 'Pearson') 
    
    XX = zeros(N,N); XY = zeros(N,1);
    for i = 1:N
        xi = X(:,i);
        for j = i:N
            xj = X(:,j);
            Corr = corr(xi,xj,'Type',method);
            XX(i,j) = Corr*median(abs(xi-median(xi)))*median(abs(xj-median(xj)));
            XX(j,i) = XX(i,j);
        end
        Corr = corr(xi,Y,'Type',method);
        XY(i) = Corr*median(abs(xi-median(xi)))*median(abs(Y-median(Y)));
    end
end

if strcmp(method, 'Shrinkage')
    XX = shrinkage_cov(X,'rblw');
    XY = shrinkage_cov([X Y],'rblw');
    XY = XY(1:end-1,end);
end


if strcmp(method, 'fastMCD')
    XX = robustcov(X);
    XY = robustcov([X,Y]);
    XY = XY(1:end-1,end);
end

if strcmp(method, 'Rho')
    XX = zeros(N,N); XY = zeros(N,1);
    for i = 1:N
        xi = X(:,i);
        for j = i:N
            xj = X(:,j);
            Corr1 = corr(xi,xj,'Type','Kendall');
            Corr2 = corr(xi,xj,'Type','Spearman');
            Corr3 = corr(xi,xj,'Type','Pearson');
            Corr = mean([Corr1,Corr2,Corr3]);
            XX(i,j) = Corr*median(abs(xi-median(xi)))*median(abs(xj-median(xj)));
            XX(j,i) = XX(i,j);
        end
        Corr1 = corr(xi,xj,'Type','Kendall');
        Corr2 = corr(xi,xj,'Type','Spearman');
        Corr3 = corr(xi,xj,'Type','Pearson');
        Corr = mean([Corr1,Corr2,Corr3]);
        XY(i) = Corr*median(abs(xi-median(xi)))*median(abs(Y-median(Y)));
    end
end

Cov.XX = XX;
Cov.XY = XY;

end