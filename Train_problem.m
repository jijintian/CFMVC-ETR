function [labels,F,Z,B,converge_Z,converge_Z_G] = Train_problem(X, cls_num,alpha,delta,gamma)
% X is a cell data, each cell is a matrix in size of d_v *N,


nV = length(X);
N = size(X{1},2);
t=cls_num;%anchors
nC=cls_num;
%% ============================ Initialization ============================
for k=1:nV
    X{k}=X{k}';
    Z{k} = zeros(N,t); 
    W{k} = zeros(N,t);
    J{k} = zeros(N,t);
    B{k} = zeros(N,t);
    A{k} = zeros(t,size(X{k},2));
    E{k} = zeros(N,size(X{k},2)); %E{2} = zeros(size(X{k},1),N);
    Y{k} = zeros(N,size(X{k},2)); %Y{2} = zeros(size(X{k},1),N);
end

w = zeros(N*t*nV,1);
j = zeros(N*t*nV,1);
sX = [N, t, nV];

Isconverg = 0;epson = 1e-7;
iter = 0;
mu = 10e-5; max_mu = 10e10; pho_mu = 2;
rho = 0.0001; max_rho = 10e12; pho_rho = 2;

converge_Z=[];
converge_Z_G=[];


%% ================================ Upadate ===============================
while(Isconverg == 0)
    %%=================Upadate B^k=========================================
H_B ={};
HH_b =[];
for i = 1 : nV 
    H_B{i} = Y{i}*A{i}'+mu*X{i}*A{i}'-mu*Z{i}-mu*E{i}*A{i}';
    HH_b(i,:) = reshape(H_B{i}, 1, N*t);
end
B_C = ones(nV,nV)*gamma - diag(ones(1,nV)*(gamma-mu));
    if det(B_C) == 0
        solution = (pinv(B_C) * HH_b);
        fprintf('------------')
    else
        solution = (B_C \ HH_b);
    end
    for i = 1 : nV 
    B{i} = reshape(solution(i,:),N,t);
    end
%% ============================== Upadate Z^k =============================
     clear i l
       temp_E =[];
      for k =1:nV;
          tmp = mu*X{k}*A{k}'-mu*E{k}*A{k}'+Y{k}*A{k}'+ rho*J{k}- W{k}+mu*B{k};
          [Zu,Zs,Zv] = svd(tmp,'econ');
           Z{k}=Zu*Zv';  
          temp_E=[temp_E,X{k}-(Z{k}+B{k})*A{k}+Y{k}/mu];
      end
      clear k 

%% =========================== Upadate E^k, Y^k ===========================
        temp_E=temp_E';
       [Econcat] = solve_l1l2(temp_E,alpha/mu);
       ro_b =0;
       E{1} =  Econcat(1:size(X{1},2),:)';
       Y{1} = Y{1} + mu*(X{1}-(Z{1}+B{1})*A{1}-E{1});
       ro_end = size(X{1},2);
       for i=2:nV
           ro_b = ro_b + size(X{i-1},2);
           ro_end = ro_end + size(X{i},2);
           E{i} =  Econcat(ro_b+1:ro_end,:)';
           Y{i} = Y{i} + mu*(X{i}-(Z{i}+B{i})*A{i}-E{i});
       end

%% ============================= Upadate J^k ==============================

                Z_tensor = cat(3, Z{:,:});
                W_tensor = cat(3, W{:,:});
                z = Z_tensor(:);
                w = W_tensor(:);
                %[g, objV] = wshrinkObj_weight(z + 1/mu*w,beta/mu,sX,0,3);
                %G_tensor = reshape(g, sX);
                J_tensor = solve_G(Z_tensor + 1/rho*W_tensor,rho,sX,delta);
                j = J_tensor(:);

%% ============================== Upadate W ===============================
        w = w + rho*(z - j);

 %% ============================== Upadate A{v} ===============================
   %G={};
for i = 1 :nV
    G{i}=(Z{i}+B{i})'*(Y{i}+mu*X{i}-mu*E{i});
    [Au,ss,Av] = svd(G{i},'econ');
        A{i}=Au*Av';
end

%% ====================== Checking Coverge Condition ======================
    max_Z=0;
    max_Z_G=0;
    Isconverg = 1;
    for k=1:nV
        if (norm(X{k}-(Z{k}+B{k})*A{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-(Z{k}+B{k})*A{k}-E{k},inf);
            Isconverg = 0;
            max_Z=max(max_Z,history.norm_Z );
        end
        
        J{k} = J_tensor(:,:,k);
        W_tensor = reshape(w, sX);
        W{k} = W_tensor(:,:,k);
        if (norm(Z{k}-J{k},inf)>epson)
            history.norm_Z_G = norm(Z{k}-J{k},inf);
            Isconverg = 0;
            max_Z_G=max(max_Z_G, history.norm_Z_G);
        end
    end
    converge_Z=[converge_Z max_Z];
    converge_Z_G=[converge_Z_G max_Z_G];
   
    
    if (iter>50)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
end
% Sbar=[];
% for i = 1:nV
%     %Zy{i} = diag(sum(Z{i},2)+eps);
%     Sbar=cat(1,Sbar,1/sqrt(nV)*Z{i}');
% end
% [F,Sig,V] = mySVD(Sbar',nC); 

Sbar=[];
for i = 1:nV
      FF = [Z{i},B{i}];
    %Zy{i} = diag(sum(Z{i},2)+eps);
    Sbar=cat(1,Sbar,FF');
end
[F,Sig,V] = mySVD(Sbar',nC); 

rand('twister',5489)
labels=litekmeans(F, nC, 'MaxIter', 100,'Replicates',10);%kmeans(U, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
end
