function [Ix,Iy] = ddplot(X,Y)
%   ddplot is a depth-depth plot
%   X,Y are multivariate data
z= [X; Y];

Ix= zeros(length(z),1);
Iy= zeros(length(z),1);

for i=1:length(z)
    cont=0;
    cont2=0;
    for j=1:length(X)
        cont= cont +  norm(z(i,:)-X(j,:));
    end
    Ix(i)=cont;
    for j=1:length(Y)
        cont2= cont2 +  norm(z(i,:)-Y(j,:));
    end
    Iy(i)=cont2;
end
clf
plot(Ix,Iy)
end
