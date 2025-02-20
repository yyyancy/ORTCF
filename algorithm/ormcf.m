function [X_recov,S,O,time] = ormcf(M_0,X,Omega,rank,iter,lambda_1,lambda_2,lambda_3)
dim_ = size(M_0);
d = size(X,2);
A = zeros([rank,rank]);
B = zeros([dim_(1),rank]);
D = randn([dim_(1),rank]);
D_old = randn([dim_(1),rank]);
R = zeros([rank,dim_(2)]);
r = zeros([rank,1]);
S = zeros(dim_);
O = zeros(dim_);
tol = 1e-7;
for i = 1: iter
    for n = 1: dim_(2)
        tic;
        m = M_0(:,n,:,:);
        %更新右张量R和噪声张量S
        omega_ = Omega(:,n,:,:);
        [s,r,o] = R_update(m,omega_,D,r,rank,lambda_1,lambda_2);
        O(:,n) = o;
        S(:,n) = s;
        R(:,n) = r;
        %更新左张量L2
        A = A + r * r';
        B = B + (m-s-o) * r';
        L = L_update(D,X,lambda_1,lambda_3);
        D = D_update(D,A,B,X,L,lambda_3);
        diff = norm(D_old(:)-D(:));
        D_old = D;
        time = toc;
    end
end
X_recov = X * L * R;

    function [s_,r_,o_] = R_update(m_,Omega_,D_,r_,rank_,lambda_1,lambda_2)
    d_ = size(m_);
    D_1 = D_' * D_ + lambda_1* eye([rank_,rank_]);
    D_1_inv = tinv(D_1);
    D_2 = tran(D_);
    D_tilde = D_1_inv * D_2;
    s_ = zeros(d_);
    s_new = s_;
    r_new = r_;
    o_ = zeros(d_);
    converged = false;
    iter_ = 0;
    while ~converged
        iter_ = iter_ +1;
        s_ = soft_threshold(m_- o_ - D_ *r_,lambda_2);
        r_ = D_tilde * (m_- s_ - o_);
        o_ = m_ - s_ - D_ * r_;
        o_ = o_ .* (1-Omega_);
        error = max([norm(s_(:)-s_new(:),"fro"),norm(r_(:)-r_new(:),"fro")]);
        if error < tol
            converged = true;
        end
        if ~converged && iter_ >= 1000
            converged = true ;    
        end
        s_new = s_;
        r_new = r_;
    end 
end

    function [L_] = L_update(D_,X,lambda_1,lambda_3)
    X_1 = tran(X) * X;
    d_ = size(X_1);
    X_2 = inv((lambda_1/lambda_3)* eye(d_)+X_1);
    X_3 = X' * D_;
    L_ = X_2 * X_3;         
end

    function [D_] = D_update(D_,A_,B_,X,L_,lambda_5)
    d_ = size(B_);
    M_ = X * L_;
    A_ = A_ + lambda_5 * eye(size(A_));
    B_ = B_ + lambda_5 * M_;
    for j = 1:d_(2)
            D_(:,j) = D_(:,j) + (B_(:,j) - D_(:,:) * A_(:,j))/A_(j,j);
    end
    end


end