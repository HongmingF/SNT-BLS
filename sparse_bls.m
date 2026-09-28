function wk = sparse_bls(Z,X,lambda,itrs)

%wk：输出变量，表示最终计算得到的稀疏权重矩阵。
%Z：输入矩阵，通常是特征矩阵。
%X：目标矩阵，通常是需要拟合的输出。
%lambda：正则化参数，用于控制稀疏性。
%itrs：迭代次数，控制优化的执行步数
%ρ=1



N1 = size(Z,2);%Z 的列数（特征数量）
d = size(X,2);%X 的列数（目标输出的维度
%初始化为零矩阵，大小为 N1×d
x = zeros(N1,d);
wk = x; %wk：最终的稀疏权重矩阵。
ok=x;%ok：临时变量，用于更新稀疏权重
uk=x;%uk：拉格朗日乘子，用于调整优化目标

L1=eye(N1)/((Z') * Z+eye(N1));%L1=(Z⊤Z+I)−1这是一个正则化逆矩阵，用于稳定求解
L2=L1*Z'*X;%L2=L1Z⊤X用于快速计算中间变量

for i = 1:itrs
    tempc=ok-uk;%临时变量，表示当前稀疏权重和拉格朗日乘子的差
    ck =  L2+L1*tempc;%结合拉格朗日乘子和之前的优化状态，计算当前的候选稀疏权重
    ok=shrinkage(ck+uk, lambda);%调用 shrinkage 函数对候选稀疏权重 ck + uk 进行稀疏化处理，使其满足稀疏约束
    uk=uk+(ck-ok);%使用拉格朗日乘子的更新公式，使其逐步逼近约束条件
    wk=ok;%保存当前的稀疏权重矩阵

end
end
%实现软阈值（Soft Thresholding）稀疏化操作，用于将小于某个阈值的值强制为 0
function z = shrinkage(a, kappa)
    z = max( a - kappa,0 ) - max( -a - kappa ,0);
end
%输入：  a：待处理矩阵。  kappa：阈值，控制稀疏性（与 lambda 相关）。
%输出： z：稀疏化后的矩阵。
%逻辑：
%如果 a>κ，输出 a−κ。
%如果 a<−κ，输出a+κ。
%如果 a∣≤κ，输出 0。




% function p = objective(A, b, lam, x, z)
%     p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
% end
% % toc

