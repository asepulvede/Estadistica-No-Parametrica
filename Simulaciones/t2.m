filename = 'return.txt';
[X,~]=importdata(filename);

%% punto 1.a
X900= X(1:500,:);

% si m es 1 provienen de distribucion diferente
m_wil= zeros(size(X900,2),size(X900,2));
for i=1:size(X900,2)
    for j=1:size(X900,2)
        p= ttest(X900(:,i)',X900(:,j)');
        if p>0.05
            m_wil(i,j)= 0 ;
        else
            m_wil(i,j)=1;
        end
    end
end
%% 

m_mww= zeros(size(X900,2),size(X900,2));
for i=1:size(X900,2)
    for j=1:size(X900,2)
        pv= mwwtest(X900(:,i)',X900(:,j)').p;
        if pv(2)>0.05
            m_mww(i,j)= 0 ;
        else
            m_mww(i,j)=1;
        end
    end
end
%% 

a1= X900(:,6);
a2= X900(:,8);

xm1 = min(a1):0.05:max(a1);
xm2 = min(a2):0.05:max(a2);

pd_kernel = fitdist(a1,'Kernel');
pdf_kernel = pdf(pd_kernel,xm1);

pd2_kernel = fitdist(a2,'Kernel');
pdf2_kernel = pdf(pd2_kernel,xm2);

clf
plot(xm1,pdf_kernel,'Color','b','LineWidth',2);
hold on
plot(xm2,pdf2_kernel,'Color','g','LineWidth',2);
%% 
cont=1;
for i=1:size(X900,2)
    a1= X900(:,i);
    for j=1:size(X900,2)
        a2= X900(:,j);
        xm1 = min(a1):0.05:max(a1);
        xm2 = min(a2):0.05:max(a2);

        pd_kernel = fitdist(a1,'Kernel');
        pdf_kernel = pdf(pd_kernel,xm1);

        pd2_kernel = fitdist(a2,'Kernel');
        pdf2_kernel = pdf(pd2_kernel,xm2);

        subplot(size(X900,2),size(X900,2),cont)
        plot(xm1,pdf_kernel,'Color','b','LineWidth',2); 
        hold on 
        plot(xm2,pdf2_kernel,'Color','r','LineWidth',2);
        cont= cont+1;
    end
end
%% 1.b
a1= X(:,1);
a2= X(:,2);
reg= fitlm(a1,a2)

%% Regresión Lineal Robusta

I= mean(A1-A2>=0)
