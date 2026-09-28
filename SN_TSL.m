function [Q, W, Omega, obj_value, H] = SN_TSL(X, c, H, alpha, beta, lambda, n_L, H_L_onehot)
% SN_TSL
% Revised implementation focused on numerical stability and peak accuracy.
%
% Input dimensions used by this code:
%   X              : d x n
%   H_L_onehot     : n_L x c
%   H              : optional n x c initialization (can be empty)
%
% Output dimensions:
%   W              : r x d
%   Q              : c x r
%   Omega          : r x n
%   H              : n x c

%% optimization settings
tol = 1e-6;
rho = 1.1;
max_mu = 1e8;
mu = 1e-3;          % more practical than 1e-5 for the current thresholding scheme
maxIter = 100;

[d, n] = size(X);
r = 3 * c;

%% basic checks
if n_L < 1 || n_L > n
    error('n_L must satisfy 1 <= n_L <= number of samples.');
end
if size(H_L_onehot, 1) ~= n_L || size(H_L_onehot, 2) ~= c
    error('H_L_onehot must be n_L-by-c.');
end

%% initialization
W = normcol_equal(randn(r, d));
Q = normcol_equal(randn(c, r));

J = zeros(r, n);
Y1 = zeros(r, n);          % Lagrange multiplier

% Initialize soft labels only ONCE.  Do not reset target labels every iteration.
if nargin < 3 || isempty(H) || ~isequal(size(H), [n, c])
    H = ones(n, c) / c;
else
    H = max(H, 0);
    row_sum = sum(H, 2);
    bad = row_sum <= eps;
    H(bad, :) = 1 / c;
    row_sum = sum(H, 2);
    H = H ./ row_sum;
end
H(1:n_L, :) = H_L_onehot;

obj_value = nan(maxIter, 1);

%% quantities that do not change during iteration
XXt_reg = X * X' + lambda * eye(d);

%% alternating optimization
iter = 0;
while iter < maxIter
    iter = iter + 1;

    % ---------------------------------------------------------------
    % 1) update Omega
    % ---------------------------------------------------------------
    M_omega = beta * (Q' * Q) + (mu + 1) * eye(r);
    RHS_omega = W * X + beta * Q' * H' + mu * J - Y1;
    Omega = M_omega \ RHS_omega;

    % Keep nonnegative representation.  This is retained intentionally
    % because the current code is optimized for empirical performance.
    Omega = max(Omega, 0);

    % ---------------------------------------------------------------
    % 2) update J: positive soft-thresholding
    % ---------------------------------------------------------------
    J = max(Omega + Y1 / mu - alpha / mu, 0);

    % ---------------------------------------------------------------
    % 3) update W
    % Avoid explicit inv() for better numerical stability.
    % ---------------------------------------------------------------
    W = (Omega * X') / XXt_reg;

    % ---------------------------------------------------------------
    % 4) update Q
    % ---------------------------------------------------------------
    M_q = beta * (Omega * Omega') + lambda * eye(r);
    Q = (beta * H' * Omega') / M_q;

    % ---------------------------------------------------------------
    % 5) update target soft labels H
    % ---------------------------------------------------------------
    V = (Q * Omega)';      % n x c

    for i = n_L + 1:n
        H(i, :) = EProjSimplex_new(V(i, :));
    end

    % Keep labeled samples exactly fixed.
    H(1:n_L, :) = H_L_onehot;

    % Numerical guard for target rows.
    if n_L < n
        H(n_L+1:end, :) = max(H(n_L+1:end, :), 0);
        target_sum = sum(H(n_L+1:end, :), 2);
        zero_rows = target_sum <= eps;
        if any(zero_rows)
            tempH = H(n_L+1:end, :);
            tempH(zero_rows, :) = 1 / c;
            H(n_L+1:end, :) = tempH;
            target_sum = sum(H(n_L+1:end, :), 2);
        end
        H(n_L+1:end, :) = H(n_L+1:end, :) ./ target_sum;
    end

    % ---------------------------------------------------------------
    % objective value (diagnostic only)
    % ---------------------------------------------------------------
    obj_value(iter) = ...
        0.5 * norm(W * X - Omega, 'fro')^2 + ...
        alpha * sum(abs(Omega(:))) + ...
        0.5 * beta * norm(Q * Omega - H', 'fro')^2 + ...
        0.5 * lambda * (norm(W, 'fro')^2 + norm(Q, 'fro')^2);

    % ---------------------------------------------------------------
    % convergence / multiplier update
    % ---------------------------------------------------------------
    leq1 = Omega - J;
    stopC = max(abs(leq1(:)));

    if ~isfinite(stopC) || any(~isfinite(Omega(:))) || any(~isfinite(Q(:))) || any(~isfinite(W(:)))
        warning('SN_TSL stopped because NaN/Inf was detected at iteration %d.', iter);
        break;
    end

    if stopC < tol
        break;
    end

    Y1 = Y1 + mu * leq1;
    mu = min(max_mu, mu * rho);
end

obj_value = obj_value(1:iter);

end


% -------------------------------------------------------------------------
% The following helper functions are kept for compatibility with the
% original file, although they are not used by the current main update.
% -------------------------------------------------------------------------
function E = solve_l1l2(W, lambda)
n = size(W, 1);
E = W;
for i = 1:n
    E(i, :) = solve_l2(W(i, :), lambda);
end
end

function x = solve_l2(w, lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw > lambda
    x = (nw - lambda) * w / nw;
else
    x = zeros(size(w));
end
end
