clear; clc;

rng(45);

acc_lib = [];
alpha_lib = [];
beta_lib = [];
lambda_lib = [];
seed_lib = [];

num_restart = 1;

for k = 15
   
    path = ['data_iv/subject', num2str(k), '/fea_session2_subject', num2str(k), '.mat'];
    load(path);
    [~, n_L] = size(fea);
    X_label = fea;
    clear path fea

    path = ['data_iv/subject', num2str(k), '/gnd_session2.mat'];
    load(path);
    Y_label = gnd;
    clear path gnd

  
    path = ['data_iv/subject', num2str(k), '/fea_session3_subject', num2str(k), '.mat'];
    load(path);
    [~, n_U] = size(fea);
    X_unlabel = fea;
    clear path fea

    path = ['data_iv/subject', num2str(k), '/gnd_session3.mat'];
    load(path);
    Y_unlabel = gnd;
    clear path gnd

 
    N1 = 10;
    Ng = 10;
    s = 0.8;

    for N2 = 100
      
        [~, X_label, Y_label, X_unlabel, Y_unlabel] = ...
            pretreat_3(X_label', Y_label, X_unlabel', Y_unlabel);

        [AX, AX_test] = bls_train(X_label', X_unlabel', s, N1, Ng, N2);
        X = [AX; AX_test]';       

        [~, n] = size(X);
        c = 4;
        H_L_onehot = onehot(Y_label, c);


        exp_grid = -10:2:4;
        alphalib = 2 .^ exp_grid;
        betalib = 2 .^ exp_grid;
        lambdalib = 2 .^ exp_grid;

        best_acc = -inf;
        best_alpha = NaN;
        best_beta = NaN;
        best_lambda = NaN;
        best_seed = NaN;
        H_predict = [];
        best_obj = [];

        for l1 = 1:length(alphalib)
            alpha = alphalib(l1);

            for l2 = 1:length(betalib)
                beta = betalib(l2);

                for l3 = 1:length(lambdalib)
                    lambda = lambdalib(l3);

                    for restart_id = 1:num_restart
                        
                        rng(restart_id);

                        H0 = ones(n, c) / c;
                        H0(1:n_L, :) = H_L_onehot;

                        [Q, W, Omega, obj_value, H] = ...
                            SN_TSL(X, c, H0, alpha, beta, lambda, n_L, H_L_onehot); 

                        H_u = H(n_L+1:n, :);
                        [~, predict_label] = max(H_u, [], 2);

                        acc = mean(predict_label(:) == Y_unlabel(:));

                        if acc > best_acc
                            best_acc = acc;
                            best_alpha = log2(alpha);
                            best_beta = log2(beta);
                            best_lambda = log2(lambda);
                            best_seed = restart_id;
                            H_predict = predict_label;
                            best_obj = obj_value;
                        end
                    end
                end
            end
        end

        acc_lib = [acc_lib; best_acc]; 
        alpha_lib = [alpha_lib; best_alpha]; 
        beta_lib = [beta_lib; best_beta]; 
        lambda_lib = [lambda_lib; best_lambda]; 
        seed_lib = [seed_lib; best_seed]; 

        fprintf(['subject%d  best_acc=%.4f, best_alpha=2^(%.1f), ' ...
                 'best_beta=2^(%.1f), best_lambda=2^(%.1f), seed=%d\n'], ...
                 k, best_acc, best_alpha, best_beta, best_lambda, best_seed);
    end
end
