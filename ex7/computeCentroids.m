function centroids = computeCentroids(X, idx, K)


% Useful variables
[m n] = size(X);
% You need to return the following variables correctly.
centroids = zeros(K, n);

%
numberofXink = zeros(K,1);
sumofXink = zeros(K,n);
for i = 1:size(idx,2)
	z = idx(i);
	numberofXink(z) += 1;
	sumofXink(z,:) += X(i,:);
end

centroids = sumofXink./numberofXink;

% =============================================================


end


