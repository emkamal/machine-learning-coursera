function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C = 0.3;
sigma = 0.1;

##C_options = [0.01 0.03 0.1 0.3 1 3 10 30];
##sigma_options = [0.01 0.03 0.1 0.3 1 3 10 30];

##cidx=1;
##sidx=1;
##lowesterr=9999;
##lowestcidx=1;
##
##for c=C_options
##  sidx=1;
##  for s=sigma_options
##    curmodel = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
##    predictions = svmPredict(curmodel, Xval);
##    curerror = mean(double(predictions ~= yval));
##    printf ("C: %d | sigma: %d | Err: %d \n", cidx, sidx, curerror);
##    if(curerror < lowesterr)
##      lowesterr = curerror;
##      lowestcidx = cidx;
##      lowestsidx = sidx;
##      printf (">> lowC: %d | lowsigma: %d | lowErr: %d \n", C_options(lowestcidx), sigma_options(lowestsidx), lowesterr);
##    endif
##    sidx = sidx+1;
##  endfor
##  cidx = cidx+1;
##endfor
##
##C = C_options(lowestcidx);
##sigma = sigma_options(lowestsidx);

##printf ("FINAL = C: %d | sigma: %d | Err: %d \n", C, sigma, lowesterr);

% =========================================================================

end
