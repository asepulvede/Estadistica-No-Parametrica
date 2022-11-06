function[LM, PD1, PD2] = TestDepthTukey(data1,data2)
[f1,~] = size(data1);
[f2,~] = size(data2);
data = [data1
    data2];
n = 10;%direcciones aleatorias
for j = 1:f1+f2
    for i = 1:f1
    I1(i) = depthTukey(data(j,:),data1,n);
    end
    D1(j) = sum(I1);
    for k = 1:f2
    I2(k) = depthTukey(data(j,:),data2,n);
    end
    D2(j) = sum(I2);
end
PD1 = 1-D1/sum(D1);
PD2 = 1-D2/sum(D2);
figure; plot(PD1,PD2,'o')
LM = fitlm(PD1,PD2);