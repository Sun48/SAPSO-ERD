clearvars; clc;
time_begin = tic;
warning('off');

% ===================== 实验参数 =====================
D = 500;          % 维度: 30 / 50 / 100
mf = 1000;       % 最大真实函数评估次数
runs = 1;        % 独立运行次数
sn1 = 1;         % 收敛曲线采样间隔
pop_size = 100;  % 种群规模
base_seed = [];  % 随机种子 (空 = 不固定)

% 基准函数: 1-Ackley, 2-Griewank, 3-Rosenbrock, 4-Ellipsoid, 5-Rastrigin,
%           6-CEC05_f10, 7-CEC05_f19
fitness_ids_all   = [1, 2, 3, 4, 5, 6, 7];
fitness_names_all = {'Ackley','Griewank','Rosenbrock','Ellipsoid','Rastrigin','CEC05_f10','CEC05_f19'};
fitness_bounds_all = [...
    -32.768, 32.768;  % Ackley
    -600.0,  600.0;   % Griewank
    -2.048,  2.048;   % Rosenbrock
    -5.12,   5.12;    % Ellipsoid
    -5.12,   5.12;    % Rastrigin
    -5.0,    5.0;     % CEC05_f10
    -5.0,    5.0];    % CEC05_f19

selected_fitness_ids = [1, 2, 3, 4, 5];  % 选择要测试的函数
enable_curve_plot = false;                % 是否绘制收敛曲线

% ===================== 初始化配置 =====================
init_config = struct();
init_config.mode = 'lhs';

% ===================== 局部搜索配置 =====================
use_local_search = true;
K = 3;
gap_ratio = 0.02;

ls_config = struct();
ls_config.mode = 'enhanced';
ls_config.scope = 'elite';
ls_config.elite_ratio = 0.40;
ls_config.adaptive_gap = true;
ls_config.enable_directional_multistep = true;
ls_config.directional_alphas = [0.50, 0.85, 1.05];

% ===================== 全局搜索配置 =====================
gs_config = struct();
gs_config.enable_adaptive_pso = true;
gs_config.w_max = 0.90;
gs_config.w_min = 0.40;
gs_config.c1_start = 2.50;
gs_config.c1_end = 0.50;
gs_config.c2_start = 0.50;
gs_config.c2_end = 2.50;
gs_config.enable_stagnation_jump = true;
gs_config.stagnation_window = 6;
gs_config.jump_ratio = 0.15;
gs_config.jump_sigma = 0.20;

% ===================== 搜索策略配置 =====================
% 仅保留核心策略，其余使用 SAPSO_ERD.m 中的默认值
search_config = struct();
search_config.enable_confidence_allocation = true;      % 策略3: 置信度分配
search_config.enable_msm_representative_replace = true; % 核心策略1: MSM代表替换
search_config.enable_batch_decorrelation = true;        % 策略2: K-means++批量去相关
search_config.batch_diversity_mode = 'kmeanspp';
search_config.enable_uncertainty_infill = true;         % 策略4: 不确定性填充
search_config.enable_hierarchical_surrogate = true;     % 层次代理模型
search_config.enable_hybrid_gpr_rbf = true;             % 混合 GPR-RBF 代理
search_config.verbose = true;
search_config.verbose_iter_interval = 1;

% 初始样本量
if D < 100
    initial_sample_size = 150;
else
    initial_sample_size = 200;
end

% ===================== 主循环 =====================
selected_fitness_ids = selected_fitness_ids(:)';
selected_fitness_ids = unique(selected_fitness_ids, 'stable');
num_funcs = numel(selected_fitness_ids);
num_snapshots = fix(mf / sn1);
all_curves = zeros(num_funcs, num_snapshots);

summary_best = zeros(num_funcs, 1);
summary_worst = zeros(num_funcs, 1);
summary_mean = zeros(num_funcs, 1);
summary_std = zeros(num_funcs, 1);

for fidx = 1:num_funcs
    func_id = selected_fitness_ids(fidx);
    meta_idx = find(fitness_ids_all == func_id, 1);
    func_name = fitness_names_all{meta_idx};
    Xmin = fitness_bounds_all(meta_idx, 1);
    Xmax = fitness_bounds_all(meta_idx, 2);
    fname = @(x) FITNESS(x, func_id);

    gsamp1 = zeros(runs, num_snapshots);

    for r = 1:runs
        if ~isempty(base_seed)
            rng(base_seed + 1000 * func_id + r, 'twister');
        end
        fitcount = 0;
        CE = zeros(mf, 2);
        gfs = zeros(1, num_snapshots);

        sam = lhsdesign(initial_sample_size, D);
        sam = Xmin + sam * (Xmax - Xmin);
        fit = zeros(1, initial_sample_size);
        for i = 1:initial_sample_size
            fit(i) = fname(sam(i, :));
            fitcount = fitcount + 1;
            CE(fitcount, :) = [fitcount, fit(i)];
            if mod(fitcount, sn1) == 0
                gfs(1, fitcount / sn1) = min(CE(1:fitcount, 2));
            end
        end

        hisx = sam;
        hisf = fit;

        [~, sidx] = sort(fit);
        sam = sam(sidx, :);
        fit = fit(sidx);
        psam = sam(1:pop_size, :);
        efit = fit(1:pop_size);

        LB = repmat(Xmin, 1, D);
        UB = repmat(Xmax, 1, D);
        gap = gap_ratio * (Xmax - Xmin);

        [~, hisx, hisf, fitcount, CE, gfs, ~, ~, ~, ~] = ...
            SAPSO_ERD(fname, D, pop_size, LB, UB, psam, efit, hisx, hisf, fitcount, mf, CE, sn1, gfs, ...
            use_local_search, K, gap, ls_config, gs_config, search_config);

        fprintf('[SAPSO-ERD][%s] Run %d/%d Best fitness: %e\n', func_name, r, runs, min(hisf));
        gsamp1(r, :) = gfs;
    end

    final_values = gsamp1(:, end);
    summary_best(fidx) = min(final_values);
    summary_worst(fidx) = max(final_values);
    summary_mean(fidx) = mean(final_values);
    summary_std(fidx) = std(final_values);
    all_curves(fidx, :) = mean(gsamp1, 1);

    fprintf('[%s] best=%e, mean=%e, std=%e\n', ...
        func_name, summary_best(fidx), summary_mean(fidx), summary_std(fidx));
end

% ===================== 结果输出 =====================
eval_calls = (sn1:sn1:mf)';
fprintf('\n===== 汇总结果 =====\n');
fprintf('%-12s %12s %12s %12s\n', 'Function', 'Best', 'Mean', 'Std');
for fidx = 1:num_funcs
    meta_idx = find(fitness_ids_all == selected_fitness_ids(fidx), 1);
    fprintf('%-12s %12.4e %12.4e %12.4e\n', ...
        fitness_names_all{meta_idx}, summary_best(fidx), summary_mean(fidx), summary_std(fidx));
end

if enable_curve_plot
    figure('Name', 'SAPSO-ERD 收敛曲线', 'Position', [100, 100, 900, 550]);
    hold on;
    colors = lines(num_funcs);
    for fidx = 1:num_funcs
        gsamp_log = log(max(all_curves(fidx, :), 1e-300));
        meta_idx = find(fitness_ids_all == selected_fitness_ids(fidx), 1);
        plot(eval_calls, gsamp_log, '.-', 'Color', colors(fidx, :), ...
            'Markersize', 3, 'DisplayName', fitness_names_all{meta_idx});
    end
    legend('Location', 'best');
    xlabel('Function Evaluation Calls');
    ylabel('Mean Fitness (ln)');
    title(sprintf('SAPSO-ERD 基准函数收敛曲线 (D=%d, mf=%d)', D, mf));
    grid on;
    hold off;
end

time_cost = toc(time_begin);
fprintf('\n总耗时 (s): %.2f\n', time_cost);
