function [best_pos,bestever] = SLPSO(d, maxgen,gmd,minerror,ghx,opts)
%  disp('SLPSO global search');
% e.g., d=20; maxfe=1000;
% d: dimensionality
% maxfe: maximal number of fitness evaluations
n = d;
%parameter setting
%parameter initiliaztion
M = 100; beta=0.01;
m = M + fix(d/10);
c3 = d/M*beta;     % 对应论文中公式2的社会影响因子
PL = zeros(m,1);

for i = 1 : m
    PL(i) = (1 - (i - 1)/m)^log(sqrt(ceil(d/M)));
end

%initialization
p = zeros(m, d); 
lu = [min(ghx); max(ghx)];

XRRmin = repmat(lu(1, :), m, 1);
XRRmax = repmat(lu(2, :), m, 1);
% rand('seed', sum(100 * clock));     % (2017.5.8)
p = XRRmin + (XRRmax - XRRmin) .* lhsdesign(m, d);
v = zeros(m,d);
bestever = 1e200;
best_pos  = zeros(1,d);

FES = 0;
gen = 0;
flag_er=0;

tic;
%% main loop
% while(FES < maxfe)
while(gen < maxgen)  
    best_old=bestever;      
    bestpos_old=best_pos;   
    %rand('state', sum(100 * clock));
    FES = FES + m;
    %[fitness bestp besty rank] = update(p, funcid);
    fitness = gmd(p);   % surrogate model: RBF network, Guassian process
%   fprintf('Best fitness: %e\n', bestever);

    %population sorting
    [fitness rank] = sort(fitness, 'descend');   % 降序排列，最后那个是最小的
    p = p(rank,:);
    v = v(rank,:);
    besty = fitness(m);
    bestp = p(m, :);
    [bestever,id] = min([besty, bestever]);
    best_new = bestever;    
    if id == 1
        best_pos = bestp;       
        bestpos_new=best_pos;   
    elseif id == 2
        best_pos = bestpos_old;
        bestpos_new=best_pos;
    end
    
    error=abs(best_old-best_new);
    if error <= minerror
        flag_er=flag_er+1;
    else
        flag_er=0;
    end
    if flag_er >=20       % 新颖的结束循环的方法，值得学习
        break;
    end
    
    %center position
    center = ones(m,1)*mean(p);
    %random matrix
    %rand('seed', sum(100 * clock));
    randco1 = rand(m, d);
    %rand('seed', sum(100 * clock));
    randco2 = rand(m, d);
    %rand('seed', sum(100 * clock));
    randco3 = rand(m, d);
    winidxmask = repmat([1:m]', [1 d]);
    winidx = winidxmask + ceil(rand(m, d).*(m - winidxmask));
    %winidx = m - floor(0.5*rand(m, d).*(m - winidxmask));
    pwin = p;
    for j = 1:d
        pwin(:,j) = p(winidx(:,j),j);
    end
    %learning
     lpmask = repmat(rand(m,1) < PL, [1 d]);
     lpmask(m,:) = 0;
     v1 =  1*(randco1.*v + randco2.*(pwin - p) + c3*randco3.*(center - p));
     p1 =  p + v1; 
     v = lpmask.*v1 + (~lpmask).*v;         
     p = lpmask.*p1 + (~lpmask).*p;
     %boundary
    for i = 1:m
        p(i,:) = max(p(i,:), lu(1,:));
        p(i,:) = min(p(i,:), lu(2,:));
    end
    gen = gen + 1;
end
end
