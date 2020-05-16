fpath='SymptomDaysTrain.mat';
a=load(fpath);
x = a.a;
% meanR = mean(x);
% stdR = std(x);

% compute mean
mx = mean(x);
% compute the standard deviation
sigma = std(x);
% compute the median
medianx = median(x);
% STEP 1 - rank the data
y = sort(x);
% compute 25th percentile (first quartile)
Q(1) = median(y(y<median(y)));
% compute 50th percentile (second quartile)
Q(2) = median(y);
% compute 75th percentile (third quartile)
Q(3) = median(y(y>median(y)));
% compute Interquartile Range (IQR)
IQR = Q(3)-Q(1);
