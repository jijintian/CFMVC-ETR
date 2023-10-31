
clear;
clc

folder_now = pwd;
addpath([folder_now, '\funs']);
addpath([folder_now, '\dataset']);
dataname=["BBCSport"];
%% ==================== Load Datatset and Normalization ===================
for it_name = 1:length(dataname)
    load(strcat('dataset/',dataname(it_name),'.mat'));
    
    gt = truelabel{1};
    cls_num=length(unique(gt));
    X=data';
    nV = length(X);
    
    for v=1:nV
        [X{v}]=NormalizeData(X{v});
    end
    
    %% ========================== Parameters Setting ==========================
    result=[];
    num = 0;
    max_val=0;
    record_num = 0;
    ii=0;
    %% ============================ Optimization ==============================
    for i = -4:1:3
      for i_gamma = -4:1:3
        for j = -4:1:0
                alpha = 10^(i);
                gamma = 10^(i_gamma);
                delta=10^(j);
                ii=ii+1;
                tic;
                [y,FF,Z,B,converge_Z,converge_Z_G] = Train_problem(X, cls_num,alpha,delta,gamma); 
                time = toc;
                [result(ii,:),res]=  ClusteringMeasure(gt, y);
                [result(ii,:),time]
                if result(ii,1) > max_val
                    max_val = result(ii,1);
                    record = [i,i_gamma,j,time];
                    record_result = result(ii,:);
                    record_c ={FF,Z,B,converge_Z,converge_Z_G};
                    record_time = time;
                end
        end
      end
    end
     save('.\result\result_'+dataname(it_name),'result','record','max_val','record_result','time')
   % save('.\result\result_'+dataname(it_name),'result','record','max_val','record_result','record_c','time')
end

