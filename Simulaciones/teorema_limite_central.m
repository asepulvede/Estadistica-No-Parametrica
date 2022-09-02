%xi de distribucion uniforme
n = 20;
for j = 1:n 
    D = rand(1000,j);
    media = mean(D,2);
    hist(media) 
    pause(2)
end

%xi de distribucion lognormal
for j = 1:n 
    D = -log(rand(1000,j));
    media = mean(D,2);
    hist(media) 
    pause(2)
end
