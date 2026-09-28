% function M = normcol_equal(M)
%     % Normalize the columns of a matrix to have unit norm
%     norms = sqrt(sum(M.^2, 1)); % Compute the norm of each column
%     norms(norms == 0) = 1; % Prevent division by zero for zero-norm columns
%     M = M ./ norms; % Normalize columns
% end


function [X] = normcol_equal(X)
    if size(X,1) == 1
        X = X./norm(X);
    else
        X = X*diag(1./sqrt(sum(X.*X)));
    end
end