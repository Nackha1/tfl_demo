function m = mcr_on_split(Br_split, phi, theta, ytrue)
% MCR_ON_SPLIT  Sets theta, monitors phi, returns misclassification rate on split.
Br_split.SetParamSpec('th', theta);
r    = Br_split.CheckSpec(phi);
yhat = double(r>=0);                % 1=normal, 0=ringing
m    = mean(yhat ~= ytrue);
end
