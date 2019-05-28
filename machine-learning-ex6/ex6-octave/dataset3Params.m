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

%% === TO CALCULATE OPTIMAL C AND SIGMA UNCOMMENT THE CODE BELOW ===
%% ========= THIS HAS BEEN COMMENTED TO SPEED UP RUNNING ===========

%CTests = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%sigmaTests = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%
%errors = zeros(size(CTests,2), size(sigmaTests,2));
%
%for i = 1:size(CTests,2)
%  for j = 1:size(sigmaTests,2)
%    model = svmTrain(X, y, CTests(i), @(x1, x2) gaussianKernel(x1, x2, sigmaTests(j)));
%    predictions = svmPredict(model, Xval);
%    errors(i, j) = mean(double(predictions ~= yval));
%  endfor
%endfor
%
%[i,j] = find(errors == min(min(errors)));
%
%C = CTests(i);
%sigma = sigmaTests(j);
%
%disp("Selected C value is:")
%disp(C)
%disp("")
%disp("Selected sigma value is:")
%disp(sigma)
%disp("")

%% ============ THESE ARE THE PARAMETERS FOUND BY THE CODE ABOVE ============
%% ================== GIVEN MANUALLY TO SPEED UP RUNNING ====================
%% === COMMENT THIS SECTION AWAY IF YOU WANT TO USE CALCULATED PARAMETERS ===

C = 1;
sigma = 0.1

% =========================================================================

end
