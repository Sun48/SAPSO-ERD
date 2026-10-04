
function [bestfit,hx,hf,fitcount,CE,gfs,pbest,pbestval,formula1,formula2]...
    = SAPSO_ERD(fname,Dimension,Particle_Number,VRmin,VRmax,pos0,e0,hx,hf,fitcount,mf,CE,sn1,gfs,use_local_search,K,gap,ls_config,gs_config,search_config)

ps = Particle_Number;
D = Dimension;

if nargin < 15 || isempty(use_local_search)
    use_local_search = false;
end
if nargin < 16 || isempty(K)
    K = 3;
end
if nargin < 17 || isempty(gap)
    gap = 0.02 * mean(VRmax - VRmin);
end
if nargin < 18 || isempty(ls_config)
    ls_config = default_ls_policy_config();
else
    ls_config = merge_cfg(default_ls_policy_config(), ls_config);
end
if nargin < 19 || isempty(gs_config)
    gs_config = default_global_policy_config();
else
    gs_config = merge_cfg(default_global_policy_config(), gs_config);
end
if nargin < 20 || isempty(search_config)
    search_config = default_search_policy_config();
else
    search_config = merge_cfg(default_search_policy_config(), search_config);
end
search_config = normalize_search_policy_config(search_config);

cc = [2.05 2.05];
iwt = 0.7298;

lower_bound = VRmin(1, :);
upper_bound = VRmax(1, :);

mv = 0.5 * (VRmax - VRmin);
VRmin = repmat(VRmin, ps, 1);
VRmax = repmat(VRmax, ps, 1);
Vmin = repmat(-mv, ps, 1);
Vmax = -Vmin;

vel = Vmin + 2 .* Vmax .* rand(ps, D);
pos = pos0;
e = e0;
pbest = pos;
pbestval = e;
[gbestval,gbestid] = min(pbestval);
gbest = pbest(gbestid,:);
gbestrep = repmat(gbest,ps,1);

if D < 100
    gs = 150;
elseif D == 100
    gs = 200;
else
    gs = 200;
end

gs = min(gs, numel(hf));

besty = 1e200;
bestp = zeros(1,D);
formula1 = 0;
formula2 = 0;
gen = 0;
stagnation_counter = 0;
last_gbestval = gbestval;
surrogate_err_ema = search_config.confidence_init_error;
surrogate_calib_scale = 1.0;
surrogate_calib_bias = 0.0;
search_config_base = search_config;
gs_config_base = gs_config;
use_local_search_base = use_local_search;

while fitcount < mf
    progress = min(1.0, fitcount / max(mf, 1));
    [search_cfg_iter, gs_cfg_iter, use_ls_iter, stage_mode] = derive_iteration_policy( ...
        search_config_base, gs_config_base, use_local_search_base, ...
        pos, lower_bound, upper_bound, progress, stagnation_counter, surrogate_err_ema);
    ls_cfg_iter = ls_config;

    [ghx, ghf] = build_training_subset(hx, hf, gs, lower_bound, upper_bound, search_cfg_iter, gbest);
    search_cfg_active = apply_landscape_guard_gate(search_cfg_iter, ghx, ghf, gbest, progress, stagnation_counter);

    surrogate_model = build_surrogate_predictor(ghx, ghf, D, search_cfg_active, lower_bound, upper_bound);
    FUN_raw = @(x) surrogate_predict_mean(surrogate_model, x);
    FUN = wrap_surrogate_predictor(FUN_raw, surrogate_calib_scale, surrogate_calib_bias, search_cfg_active);

    besty_old = besty;
    bestp_old = bestp;

    progress_sl = progress;
    maxgen = get_slpso_maxgen(D, progress_sl, stagnation_counter, search_cfg_active);
    slpso_starts = get_slpso_starts(search_cfg_active);
    minerror = 1e-6;
    slpso_learning_opts = build_slpso_learning_options(search_cfg_active, progress_sl, stagnation_counter);

    bestp_pred = zeros(1, D);
    besty_pred = inf;
    for sid = 1:slpso_starts
        [cand_bestp, cand_besty] = SLPSO(D, maxgen, FUN, minerror, ghx, slpso_learning_opts);
        if cand_besty < besty_pred
            besty_pred = cand_besty;
            bestp_pred = cand_bestp;
        end
    end
    bestp = bestp_pred;

    besty = feval(fname,bestp);
    [surrogate_calib_scale, surrogate_calib_bias] = update_surrogate_calibration(...
        surrogate_calib_scale, surrogate_calib_bias, besty_pred, besty, hf, search_cfg_active);
    fitcount = fitcount + 1;
    if fitcount <= mf
        CE(fitcount,:) = [fitcount,besty];
        if mod(fitcount,sn1) == 0
            cs1 = fitcount/sn1;
            gfs(1,cs1) = min(CE(1:fitcount,2));
        end
    end

    besty_new = besty;
    bestp_new = bestp;
    if besty_new < besty_old
        besty = besty_new;
        bestp = bestp_new;
        bestprep = repmat(bestp_new,ps,1);
    else
        besty = besty_old;
        bestp = bestp_old;
        bestprep = repmat(bestp_old,ps,1);
    end

    [hx, hf, ~] = upsert_history_point(hx, hf, bestp, besty, lower_bound, upper_bound, search_cfg_iter);

    if ghf(end) > besty
        [ghx, ghf] = build_training_subset(hx, hf, gs, lower_bound, upper_bound, search_cfg_active, gbest);
        surrogate_model = build_surrogate_predictor(ghx, ghf, D, search_cfg_active, lower_bound, upper_bound);
        FUN_raw = @(x) surrogate_predict_mean(surrogate_model, x);
        FUN = wrap_surrogate_predictor(FUN_raw, surrogate_calib_scale, surrogate_calib_bias, search_cfg_active);
    end

    if gs_cfg_iter.enable_adaptive_pso
        iwt_curr = gs_cfg_iter.w_max - (gs_cfg_iter.w_max - gs_cfg_iter.w_min) * progress;
        c1_curr = gs_cfg_iter.c1_start + (gs_cfg_iter.c1_end - gs_cfg_iter.c1_start) * progress;
        c2_curr = gs_cfg_iter.c2_start + (gs_cfg_iter.c2_end - gs_cfg_iter.c2_start) * progress;
    else
        iwt_curr = iwt;
        c1_curr = cc(1);
        c2_curr = cc(2);
    end

    aa = c1_curr.*rand(ps,D).*(pbest-pos) + c2_curr.*rand(ps,D).*(gbestrep-pos);
    w_vec = iwt_curr * ones(ps,1);

    if besty < gbestval
        [~,ip,~] = intersect(pbest,gbest,'rows');
        pbest(ip,:) = bestp;
        pbestval(ip) = besty;
        gbestrep = bestprep;
        formula1 = formula1 + 1;
    else
        formula2 = formula2 + 1;
    end

    vel = repmat(w_vec, 1, D).*(vel+aa);
    vel = (vel>Vmax).*Vmax + (vel<=Vmax).*vel;
    vel = (vel<Vmin).*Vmin + (vel>=Vmin).*vel;

    pos = pos + vel;
    pos = ((pos>=VRmin)&(pos<=VRmax)).*pos ...
        + (pos<VRmin).*(VRmin + 0.25.*(VRmax-VRmin).*rand(ps,D)) ...
        + (pos>VRmax).*(VRmax - 0.25.*(VRmax-VRmin).*rand(ps,D));

    if gs_cfg_iter.enable_stagnation_jump && stagnation_counter >= gs_cfg_iter.stagnation_window
        [pos, vel] = apply_stagnation_jump(pos, vel, gbest, pbestval, lower_bound, upper_bound, gs_cfg_iter);
        stagnation_counter = 0;
    end

    e = reshape(FUN(pos), 1, []);
    if use_ls_iter
        [apply_ls, gap_scale] = get_dynamic_local_search_params(ls_cfg_iter, progress, stagnation_counter, gen);
        if apply_ls
            [pos, e] = apply_local_search_policy(pos, e, FUN, D, gap * gap_scale, K, lower_bound, upper_bound, ls_cfg_iter);
        end
    end

    unc_model_all = surrogate_predict_uncertainty(surrogate_model, pos);
    [candidx, pos_trmem] = select_prescreen_candidates(...
        pos, e, pbestval, lower_bound, upper_bound, search_cfg_active, hx, hf, ghx, surrogate_err_ema, unc_model_all);

    [~,ih,ip] = intersect(hx,pos_trmem,'rows');
    if ~isempty(ip)
        pos_trmem(ip,:) = [];
        e(candidx(ip)) = hf(ih);
        candidx(ip) = [];
    end

    ssk = size(pos_trmem,1);
    pred_trmem = e(candidx);
    e_trmem = evaluate_exact_batch(fname, pos_trmem, search_cfg_active);
    for k = 1:ssk
        fitcount = fitcount + 1;
        if fitcount <= mf
            CE(fitcount,:) = [fitcount,e_trmem(k)];
            if mod(fitcount,sn1) == 0
                cs1 = fitcount/sn1;
                gfs(1,cs1) = min(CE(1:fitcount,2));
            end
        end

        [hx, hf, ~] = upsert_history_point(hx, hf, pos_trmem(k,:), e_trmem(k), lower_bound, upper_bound, search_cfg_iter);

        kp = candidx(k);
        if e_trmem(k) < pbestval(kp)
            pbest(kp,:) = pos_trmem(k,:);
            pbestval(kp) = e_trmem(k);
        end
    end
    surrogate_err_ema = update_surrogate_error_ema(surrogate_err_ema, pred_trmem, e_trmem, search_cfg_active);
    [surrogate_calib_scale, surrogate_calib_bias] = update_surrogate_calibration(...
        surrogate_calib_scale, surrogate_calib_bias, pred_trmem, e_trmem, hf, search_cfg_active);

    [gbestval,tmp] = min(pbestval);
    gbest = pbest(tmp,:);
    gbestrep = repmat(gbest,ps,1);
    bestfit = min([gbestval,besty]);

    if gbestval < last_gbestval - gs_cfg_iter.improve_tol
        stagnation_counter = 0;
    else
        stagnation_counter = stagnation_counter + 1;
    end
    last_gbestval = gbestval;

    gen = gen + 1;
    do_verbose = ~isfield(search_cfg_iter, 'verbose') || search_cfg_iter.verbose;
    if do_verbose
        iter_interval = 1;
        if isfield(search_cfg_iter, 'verbose_iter_interval')
            iter_interval = max(1, round(search_cfg_iter.verbose_iter_interval));
        end
        if gen == 1 || mod(gen, iter_interval) == 0 || fitcount >= mf
            fprintf(1,'Iteration: %d,  No.evaluation: %d,  Best: %e,  No.prescreen data: %d, stage: %s\n',gen,fitcount,bestfit,ssk,stage_mode);
        end
    end
end

end

function cfg = default_ls_policy_config()
cfg.mode = 'enhanced';
cfg.scope = 'elite';
cfg.elite_ratio = 0.4;
cfg.acceptance_only_improved = true;
cfg.accept_eps = 1e-12;
cfg.adaptive_gap = true;
cfg.min_gap_scale = 0.25;
cfg.diversity_ref = 0.20;
cfg.enable_directional_multistep = false;
cfg.directional_alphas = [0.5, 0.9, 1.2];
cfg.directional_max_diversity = 0.14;

cfg.center_pull_fine = 0.45;
cfg.center_pull_coarse = 0.25;
cfg.noise_ratio_fine = 0.08;
cfg.noise_ratio_coarse = 0.15;
cfg.use_levy = true;
cfg.levy_prob = 0.2;
cfg.levy_beta = 1.5;
cfg.levy_scale_fine = 0.05;
cfg.levy_scale_coarse = 0.10;

cfg.enable_dynamic_trigger = false;
cfg.dynamic_stagnation_threshold = 3;
cfg.dynamic_late_progress = 0.55;
cfg.dynamic_interval = 2;
cfg.dynamic_gap_min_scale = 0.70;
cfg.dynamic_gap_max_scale = 1.60;
cfg.dynamic_gap_stagnation_gain = 0.25;
cfg.dynamic_gap_progress_gain = 0.15;
end

function [pos_out, e_out] = apply_local_search_policy(pos_in, e_in, FUN, D, gap, K, LB, UB, cfg)
pos_out = pos_in;
e_base = reshape(e_in, 1, []);
e_out = e_base;

n = size(pos_in, 1);
if n == 0
    return;
end

idx = 1:n;
if isfield(cfg, 'scope') && strcmpi(cfg.scope, 'elite')
    elite_ratio = max(0.05, min(1.0, cfg.elite_ratio));
    elite_count = max(1, ceil(elite_ratio * n));
    [~, order] = sort(e_base, 'ascend');
    idx = order(1:elite_count);
end

domain = max(UB - LB, 1e-12);
diversity = mean(std(pos_in, 0, 1) ./ domain);
gap_eff = gap;
if isfield(cfg, 'adaptive_gap') && cfg.adaptive_gap
    gap_scale = diversity / max(cfg.diversity_ref, 1e-12);
    gap_scale = min(1.0, max(cfg.min_gap_scale, gap_scale));
    gap_eff = gap * gap_scale;
end

pos_trial = pos_in;
pos_trial(idx, :) = SAPSO_ERD_LocalSearch(pos_in(idx, :), e_base(idx), D, gap_eff, K, LB, UB, cfg);
use_directional = false;
if isfield(cfg, 'enable_directional_multistep') && cfg.enable_directional_multistep
    use_directional = true;
    if isfield(cfg, 'directional_max_diversity') && cfg.directional_max_diversity > 0
        domain = max(UB - LB, 1e-12);
        diversity_now = mean(std(pos_in, 0, 1) ./ domain);
        use_directional = diversity_now <= cfg.directional_max_diversity;
    end
end
if use_directional
    [pos_best_idx, e_best_idx] = directional_refine_along_delta( ...
        pos_in, pos_trial, idx, FUN, LB, UB, cfg);
    pos_trial(idx, :) = pos_best_idx;
    e_trial = e_base;
    e_trial(idx) = e_best_idx;
else
    e_trial = reshape(FUN(pos_trial), 1, []);
end

if isfield(cfg, 'acceptance_only_improved') && cfg.acceptance_only_improved
    eps_accept = cfg.accept_eps;
    improve_mask = (e_trial + eps_accept) < e_base;
    pos_out(improve_mask, :) = pos_trial(improve_mask, :);
    e_out(improve_mask) = e_trial(improve_mask);
else
    pos_out = pos_trial;
    e_out = e_trial;
end
end

function [best_pos_idx, best_e_idx] = directional_refine_along_delta(base_pos, trial_pos, idx, FUN, LB, UB, cfg)
best_pos_idx = trial_pos(idx, :);
if isempty(idx)
    best_e_idx = zeros(1, 0);
    return;
end
best_e_idx = reshape(FUN(best_pos_idx), 1, []);

delta = trial_pos(idx, :) - base_pos(idx, :);
if ~any(abs(delta(:)) > 0)
    return;
end

if isfield(cfg, 'directional_alphas') && ~isempty(cfg.directional_alphas)
    alpha_vec = reshape(cfg.directional_alphas, 1, []);
else
    alpha_vec = [0.5, 0.9, 1.2];
end
alpha_vec = unique(alpha_vec, 'stable');

LBm = repmat(LB, numel(idx), 1);
UBm = repmat(UB, numel(idx), 1);
base_idx = base_pos(idx, :);

for a = alpha_vec
    cand = base_idx + a .* delta;
    cand = min(max(cand, LBm), UBm);
    e_cand = reshape(FUN(cand), 1, []);
    better = e_cand < best_e_idx;
    if any(better)
        best_pos_idx(better, :) = cand(better, :);
        best_e_idx(better) = e_cand(better);
    end
end
end

function cfg = default_global_policy_config()
cfg.enable_adaptive_pso = false;
cfg.w_max = 0.90;
cfg.w_min = 0.40;
cfg.c1_start = 2.50;
cfg.c1_end = 0.50;
cfg.c2_start = 0.50;
cfg.c2_end = 2.50;

cfg.enable_stagnation_jump = false;
cfg.stagnation_window = 8;
cfg.jump_ratio = 0.20;
cfg.jump_sigma = 0.15;
cfg.jump_diff_scale = 0.0;
cfg.improve_tol = 1e-12;
cfg.enable_de_assist = false;
cfg.de_assist_ratio = 0.20;
cfg.de_assist_f = 0.55;
cfg.de_assist_cr = 0.70;
cfg.de_assist_best_pull = 0.20;

cfg.enable_island_mode = false;
cfg.num_islands = 2;
cfg.island_explore_c1_scale = 1.15;
cfg.island_explore_c2_scale = 0.85;
cfg.island_explore_w_scale = 1.05;
cfg.island_exploit_c1_scale = 0.85;
cfg.island_exploit_c2_scale = 1.15;
cfg.island_exploit_w_scale = 0.95;
cfg.migration_interval = 5;
end

function cfg = default_search_policy_config()
% Default configuration: all core strategies enabled.
cfg.candidate_max_count_ratio = 1.0;
cfg.fallback_legacy_if_empty = true;

% Dual threshold (disabled by default)
cfg.enable_dual_threshold = false;
cfg.improve_delta_ratio = 0.01;
cfg.min_improve_abs = 1e-8;

% Diversity filter (disabled by default)
cfg.enable_diversity_filter = false;
cfg.diversity_min_dist_ratio = 0.03;

% Acquisition ranking (heuristic method)
cfg.enable_acquisition_ranking = true;
cfg.acq_explore_weight = 0.25;
cfg.acq_keep_ratio = 0.75;
cfg.acq_method = 'heuristic';
cfg.acq_model_uncertainty_weight = 0.55;
cfg.ei_xi = 0.01;
cfg.lcb_kappa = 2.0;
cfg.lcb_kappa_min = 0.5;
cfg.lcb_kappa_max = 5.0;
cfg.acq_sigma_floor = 1e-8;
cfg.acq_sigma_scale = 1.0;

% Hybrid GPR-RBF surrogate (RBF prediction + GPR uncertainty)
cfg.enable_hybrid_gpr_rbf = true;
cfg.gpr_enable_optimize = true;
cfg.gpr_length_scale_init = 1.0;
cfg.gpr_signal_var_init = 1.0;
cfg.gpr_noise_var_init = 1e-4;
cfg.gpr_min_samples = 5;
cfg.gpr_ls_scale_by_dim = true;
cfg.gpr_hybrid_prediction = 'rbf';
cfg.gpr_uncertainty_only = true;

% Confidence allocation (策略3)
cfg.enable_confidence_allocation = true;
cfg.confidence_error_target = 0.12;
cfg.confidence_init_error = 0.15;
cfg.confidence_ema_alpha = 0.25;
cfg.confidence_explore_gain = 0.45;
cfg.confidence_keep_gain = 0.25;
cfg.confidence_explore_min = 0.05;
cfg.confidence_explore_max = 0.55;
cfg.confidence_keep_min = 0.45;
cfg.confidence_keep_max = 0.95;
cfg.confidence_uncertainty_weight = 0.12;
cfg.confidence_err_scale_floor = 1e-8;

% Surrogate guard
cfg.enable_surrogate_guard = true;
cfg.guard_err_high = 0.14;
cfg.guard_err_low = 0.07;
cfg.guard_min_keep_ratio = 0.72;
cfg.guard_inject_ratio = 0.12;
cfg.guard_max_inject = 4;
cfg.guard_diversity_bias = 0.30;

% Surrogate calibration (disabled by default)
cfg.enable_surrogate_calibration = false;
cfg.calib_eps = 1e-8;
cfg.calib_scale_min = 0.5;
cfg.calib_scale_max = 2.0;
cfg.calib_bias_clip_ratio = 0.1;
cfg.calib_alpha = 0.25;

% Batch decorrelation via K-means++ (策略2)
cfg.enable_batch_decorrelation = true;
cfg.batch_min_dist_ratio = 0.025;
cfg.batch_min_keep_ratio = 0.75;
cfg.batch_diversity_mode = 'kmeanspp';
cfg.batch_kmeanspp_proj_dim = 24;
cfg.batch_kmeanspp_quality_bias = 0.55;
cfg.batch_kmeanspp_refine_iters = 2;
cfg.batch_kmeanspp_rank_tradeoff = 0.20;
cfg.batch_kmeanspp_greedy_pick = true;

% MSM representative replacement (核心策略1)
cfg.enable_msm_representative_replace = true;
cfg.msm_rep_pool_top_ratio = 0.35;
cfg.msm_rep_pool_top_min = 3;
cfg.msm_rep_pool_top_max = 12;
cfg.msm_rep_random_top_min = 1;
cfg.msm_rep_keep_ratio = 0.45;

% History deduplication
cfg.enable_history_dedup = true;
cfg.history_dedup_tol_ratio = 8e-5;

% Uncertainty infill (策略4)
cfg.enable_uncertainty_infill = true;
cfg.uncertainty_infill_ratio = 0.12;
cfg.uncertainty_infill_min = 1;
cfg.uncertainty_infill_max = 3;
cfg.uncertainty_infill_model_weight = 0.75;

% Landscape guard gate
cfg.enable_landscape_guard_gate = true;
cfg.landscape_gate_rugged_corr_max = 0.35;
cfg.landscape_gate_min_progress = 0.25;
cfg.landscape_gate_min_stagnation = 2;
cfg.landscape_gate_reduce_explore_weight = 0.55;
cfg.landscape_gate_reduce_keep_ratio = 0.95;

% Hierarchical surrogate
cfg.enable_hierarchical_surrogate = true;
cfg.hier_num_local_models = 3;
cfg.hier_local_min_points = 10;
cfg.hier_kmeans_iters = 8;
cfg.hier_collab_mode = 'uncertainty_gate';
cfg.hier_uncertainty_alpha = 0.35;
cfg.hier_use_local_for_mean = false;
cfg.ensemble_global_spr_scale = 1.0;
cfg.ensemble_local_spr_scale = 0.75;
cfg.ensemble_local_ratio = 0.4;
cfg.ensemble_local_min_points = 15;
cfg.ensemble_global_weight = 0.65;
cfg.ensemble_local_weight = 0.35;
cfg.enable_surrogate_ensemble = false;

% Input normalization
cfg.enable_input_normalization = true;
cfg.input_norm_clip = true;

% RBF training stabilizer
cfg.enable_rbf_train_stabilizer = true;
cfg.rbf_stab_min_dist_ratio = 0.004;
cfg.rbf_stab_retry_dist_scale = 2.5;
cfg.rbf_stab_max_points = 100;
cfg.rbf_stab_elite_ratio = 0.35;
cfg.rbf_stab_min_points = 14;
cfg.rbf_stab_retry_spr_scale = 0.75;
cfg.rbf_stab_rcond_threshold = 1e-10;
cfg.rbf_stab_spread_min_factor = 0.35;
cfg.rbf_stab_spread_max_factor = 2.50;

% Adaptive SLPSO
cfg.enable_adaptive_slpso = true;
cfg.slpso_maxgen_min_ratio = 15;
cfg.slpso_maxgen_max_ratio = 45;
cfg.slpso_stagnation_boost_ratio = 1.25;
cfg.slpso_stagnation_window = 6;
cfg.enable_slpso_multistart = true;
cfg.slpso_starts = 2;
cfg.enable_adaptive_slpso_learning = true;
cfg.slpso_pl_scale_start = 0.8;
cfg.slpso_pl_scale_end = 1.20;
cfg.slpso_pl_stagnation_boost = 0.30;
cfg.slpso_pl_min = 0.05;
cfg.slpso_pl_max = 0.98;

% Verbose
cfg.verbose = true;
cfg.verbose_iter_interval = 1;

% Parallel exact evaluation (disabled by default)
cfg.enable_parallel_exact_eval = false;
cfg.parallel_min_batch = 8;
end

function cfg = normalize_search_policy_config(cfg)
if isfield(cfg, 'enable_msm_diverse_sampling') && ~isfield(cfg, 'enable_msm_representative_replace')
    cfg.enable_msm_representative_replace = cfg.enable_msm_diverse_sampling;
end
if ~isfield(cfg, 'enable_msm_representative_replace')
    cfg.enable_msm_representative_replace = false;
end

if isfield(cfg, 'msm_mode') && ~isempty(cfg.msm_mode)
    mode = lower(char(cfg.msm_mode));
    if strcmp(mode, 'replace')
        cfg.enable_msm_representative_replace = true;
    end
end

if isfield(cfg, 'msm_pool_top_ratio') && ~isfield(cfg, 'msm_rep_pool_top_ratio')
    cfg.msm_rep_pool_top_ratio = cfg.msm_pool_top_ratio;
end
if isfield(cfg, 'msm_pool_top_min') && ~isfield(cfg, 'msm_rep_pool_top_min')
    cfg.msm_rep_pool_top_min = cfg.msm_pool_top_min;
end
if isfield(cfg, 'msm_pool_top_max') && ~isfield(cfg, 'msm_rep_pool_top_max')
    cfg.msm_rep_pool_top_max = cfg.msm_pool_top_max;
end
if isfield(cfg, 'msm_random_top_min') && ~isfield(cfg, 'msm_rep_random_top_min')
    cfg.msm_rep_random_top_min = cfg.msm_random_top_min;
end
if isfield(cfg, 'msm_replace_keep_ratio') && ~isfield(cfg, 'msm_rep_keep_ratio')
    cfg.msm_rep_keep_ratio = cfg.msm_replace_keep_ratio;
end

if ~isfield(cfg, 'msm_rep_pool_top_ratio')
    cfg.msm_rep_pool_top_ratio = 0.35;
end
if ~isfield(cfg, 'msm_rep_pool_top_min')
    cfg.msm_rep_pool_top_min = 3;
end
if ~isfield(cfg, 'msm_rep_pool_top_max')
    cfg.msm_rep_pool_top_max = 12;
end
if ~isfield(cfg, 'msm_rep_random_top_min')
    cfg.msm_rep_random_top_min = 1;
end
if ~isfield(cfg, 'msm_rep_keep_ratio')
    cfg.msm_rep_keep_ratio = 0.45;
end
end

function [pos_out, vel_out] = apply_stagnation_jump(pos_in, vel_in, gbest, pbestval, LB, UB, cfg)
pos_out = pos_in;
vel_out = vel_in;

[ps, D] = size(pos_in);
num_jump = max(1, round(cfg.jump_ratio * ps));
[~, ord_worst] = sort(pbestval, 'descend');
jump_idx = ord_worst(1:num_jump);

domain = UB - LB;
sigma_vec = cfg.jump_sigma .* domain;
base = repmat(gbest, num_jump, 1);
noise = randn(num_jump, D) .* repmat(sigma_vec, num_jump, 1);
new_pos = base + noise;
if isfield(cfg, 'jump_diff_scale') && cfg.jump_diff_scale > 0
    rid1 = randi(ps, num_jump, 1);
    rid2 = randi(ps, num_jump, 1);
    diff_term = cfg.jump_diff_scale .* (pos_in(rid1, :) - pos_in(rid2, :));
    new_pos = new_pos + diff_term;
end
new_pos = max(new_pos, repmat(LB, num_jump, 1));
new_pos = min(new_pos, repmat(UB, num_jump, 1));

pos_out(jump_idx, :) = new_pos;
vel_out(jump_idx, :) = 0;
end

function maxgen = get_slpso_maxgen(D, progress, stagnation_counter, cfg)
max_high = max(1, round(cfg.slpso_maxgen_max_ratio * D));
if ~cfg.enable_adaptive_slpso
    maxgen = max_high;
    return;
end

max_low = max(1, round(cfg.slpso_maxgen_min_ratio * D));
maxgen = round(max_high - (max_high - max_low) * progress);
if stagnation_counter >= cfg.slpso_stagnation_window
    maxgen = round(maxgen * cfg.slpso_stagnation_boost_ratio);
end
maxgen = max(max_low, min(maxgen, round(1.5 * max_high)));
end

function starts = get_slpso_starts(cfg)
if ~cfg.enable_slpso_multistart
    starts = 1;
else
    starts = max(1, round(cfg.slpso_starts));
end
end

function opts = build_slpso_learning_options(cfg, progress, stagnation_counter)
opts = struct();
opts.enable_adaptive_learning = false;
if ~isfield(cfg, 'enable_adaptive_slpso_learning') || ~cfg.enable_adaptive_slpso_learning
    return;
end

opts.enable_adaptive_learning = true;
opts.progress = clamp_scalar(progress, 0, 1);
stag_win = max(1, round(cfg.slpso_stagnation_window));
opts.stagnation_norm = min(1.0, stagnation_counter / stag_win);
opts.pl_scale_start = cfg.slpso_pl_scale_start;
opts.pl_scale_end = cfg.slpso_pl_scale_end;
opts.pl_stagnation_boost = cfg.slpso_pl_stagnation_boost;
opts.pl_min = cfg.slpso_pl_min;
opts.pl_max = cfg.slpso_pl_max;
end

function [candidx, pos_trmem, info] = select_prescreen_candidates(pos, e, pbestval, LB, UB, cfg, hx, hf, ghx, surrogate_err_ema, model_unc_all)
e = reshape(e, 1, []);
pbestval = reshape(pbestval, 1, []);
legacy_mask = e < pbestval;
if nargin < 9 || isempty(ghx)
    ghx = hx;
end
if nargin < 10 || isempty(surrogate_err_ema)
    surrogate_err_ema = cfg.confidence_init_error;
end
if nargin < 11 || isempty(model_unc_all)
    model_unc_all = zeros(1, size(pos, 1));
else
    model_unc_all = reshape(model_unc_all, 1, []);
    if numel(model_unc_all) ~= size(pos,1)
        model_unc_all = zeros(1, size(pos,1));
    end
end
[effective_explore_weight, effective_keep_ratio] = ...
    effective_confidence_controls(cfg, surrogate_err_ema);
info = struct();
info.ConfiguredKeepRatio = cfg.acq_keep_ratio;
info.EffectiveKeepRatio = effective_keep_ratio;
info.ExplorationWeight = effective_explore_weight;
info.InitialCandidateCount = 0;
info.AfterAcquisitionCount = 0;
info.AfterGuardCount = 0;
info.AfterUncertaintyCount = 0;
info.AfterRepresentativeCount = 0;
info.FinalSelectedCount = 0;
info.MeanModelUncertainty = mean(model_unc_all, 'omitnan');
info.MeanSelectedUncertainty = NaN;
% Mechanism-specific diagnostics. These fields are observational only and
% must not alter candidate order or consume random numbers.
info.ERCAActive = double(cfg.enable_confidence_allocation);
info.RQCIAddedCount = 0;
info.RQCIMeanAddedUncertainty = NaN;
info.RQCIMeanAddedScore = NaN;
info.ECRSApplied = 0;
info.ECRSRepresentativeRank = NaN;
info.ECRSRepresentativenessGain = NaN;
info.PQDBDApplied = 0;
info.PQDBDMeanDistanceBefore = NaN;
info.PQDBDMeanDistanceAfter = NaN;
info.PQDBDDiversityGain = NaN;
info.PQDBDMinDistanceGain = NaN;

if cfg.enable_dual_threshold
    delta = max(cfg.min_improve_abs, cfg.improve_delta_ratio .* (abs(pbestval) + 1));
    mask = e <= (pbestval - delta);
else
    mask = legacy_mask;
end

candidx = find(mask);
if isempty(candidx) && cfg.fallback_legacy_if_empty
    candidx = find(legacy_mask);
end
if isempty(candidx)
    pos_trmem = zeros(0, size(pos,2));
    return;
end

[~, ord] = sort(e(candidx), 'ascend');
candidx = candidx(ord);

ratio = cfg.candidate_max_count_ratio;
if ratio < 1.0
    max_count = max(1, round(size(pos,1) * ratio));
    candidx = candidx(1:min(end, max_count));
end
info.InitialCandidateCount = numel(candidx);

if cfg.enable_diversity_filter && numel(candidx) > 1
    domain = UB - LB;
    min_dist = cfg.diversity_min_dist_ratio * norm(domain) / sqrt(max(1, numel(domain)));
    selected = zeros(1, numel(candidx));
    selected_count = 0;
    for ii = 1:numel(candidx)
        idx = candidx(ii);
        if selected_count == 0
            selected_count = 1;
            selected(selected_count) = idx;
            continue;
        end
        prev_idx = selected(1:selected_count);
        dists = vecnorm(pos(prev_idx, :) - pos(idx, :), 2, 2);
        if min(dists) >= min_dist
            selected_count = selected_count + 1;
            selected(selected_count) = idx;
        end
    end
    candidx = selected(1:selected_count);
    if isempty(candidx) && cfg.fallback_legacy_if_empty
        [~, best_ord] = min(e);
        candidx = best_ord;
    end
end

if cfg.enable_acquisition_ranking && ~isempty(candidx)
    keep_ratio = cfg.acq_keep_ratio;
    signed_factor = 0.0;
    if cfg.enable_confidence_allocation
        signed_factor = (surrogate_err_ema - cfg.confidence_error_target) / max(cfg.confidence_error_target, 1e-12);
        signed_factor = max(-1.0, min(2.0, signed_factor));
        keep_ratio = clamp_scalar(...
            cfg.acq_keep_ratio * (1 + cfg.confidence_keep_gain * signed_factor), ...
            cfg.confidence_keep_min, cfg.confidence_keep_max);
    end

    e_sel = e(candidx);
    [dist_norm, ~] = compute_min_distance_norm(pos, candidx, hx);
    [unc_norm, ~] = compute_min_distance_norm(pos, candidx, ghx);
    model_unc_norm = normalize_vector(model_unc_all(candidx));
    unc_mix = (1 - cfg.acq_model_uncertainty_weight) .* unc_norm + cfg.acq_model_uncertainty_weight .* model_unc_norm;
    method = 'heuristic';
    if isfield(cfg, 'acq_method') && ~isempty(cfg.acq_method)
        method = lower(string(cfg.acq_method));
    end

    switch char(method)
        case 'ei'
            sigma = build_sigma_from_uncertainty(unc_mix, hf, cfg);
            if isempty(hf)
                f_best = min(e_sel);
            else
                f_best = min(hf);
            end
            xi = max(0, cfg.ei_xi * (1 + 0.5 * cfg.confidence_explore_gain * signed_factor));
            imp = f_best - e_sel - xi;
            z = imp ./ max(sigma, 1e-12);
            cdf = 0.5 .* (1 + erf(z ./ sqrt(2)));
            pdf = exp(-0.5 .* (z.^2)) ./ sqrt(2 * pi);
            score_ei = imp .* cdf + sigma .* pdf;
            [~, ord2] = sort(score_ei, 'descend');
        case 'lcb'
            sigma = build_sigma_from_uncertainty(unc_mix, hf, cfg);
            kappa = cfg.lcb_kappa * (1 + 0.5 * cfg.confidence_explore_gain * signed_factor);
            kappa = clamp_scalar(kappa, cfg.lcb_kappa_min, cfg.lcb_kappa_max);
            score_lcb = e_sel - kappa .* sigma;
            [~, ord2] = sort(score_lcb, 'ascend');
        otherwise
            explore_weight = cfg.acq_explore_weight;
            uncertainty_weight = 0;
            if cfg.enable_confidence_allocation
                explore_weight = clamp_scalar(...
                    cfg.acq_explore_weight * (1 + cfg.confidence_explore_gain * signed_factor), ...
                    cfg.confidence_explore_min, cfg.confidence_explore_max);
                uncertainty_weight = cfg.confidence_uncertainty_weight;
            end

            min_e = min(e_sel);
            max_e = max(e_sel);
            e_norm = (e_sel - min_e) ./ max(max_e - min_e, 1e-12);
            score_h = e_norm - explore_weight .* dist_norm - uncertainty_weight .* unc_mix;
            [~, ord2] = sort(score_h, 'ascend');
    end

    candidx = candidx(ord2);
    keep_num = max(1, round(keep_ratio * numel(candidx)));
    candidx = candidx(1:keep_num);
end
info.AfterAcquisitionCount = numel(candidx);

if cfg.enable_surrogate_guard && ~isempty(candidx)
    if surrogate_err_ema >= cfg.guard_err_high
        inject_n = max(1, round(cfg.guard_inject_ratio * size(pos,1)));
        inject_n = min(cfg.guard_max_inject, inject_n);
        pool = setdiff(1:size(pos,1), candidx, 'stable');
        if ~isempty(pool) && inject_n > 0
            unc_pool = normalize_vector(model_unc_all(pool));
            [dist_pool, ~] = compute_min_distance_norm(pos, pool, hx);
            e_pool = normalize_vector(e(pool));
            score_guard = (1 - cfg.guard_diversity_bias) .* unc_pool ...
                + cfg.guard_diversity_bias .* dist_pool ...
                - 0.15 .* e_pool;
            [~, ord_guard] = sort(score_guard, 'descend');
            add_idx = pool(ord_guard(1:min(inject_n, numel(pool))));
            candidx = unique([candidx, add_idx], 'stable');
        end
    elseif surrogate_err_ema <= cfg.guard_err_low
        keep_n = max(1, round(cfg.guard_min_keep_ratio * numel(candidx)));
        candidx = candidx(1:keep_n);
    end
end
info.AfterGuardCount = numel(candidx);

if cfg.enable_uncertainty_infill
    [candidx, rqci_info] = apply_uncertainty_infill_indices(candidx, e, model_unc_all, cfg);
    info.RQCIAddedCount = rqci_info.AddedCount;
    info.RQCIMeanAddedUncertainty = rqci_info.MeanAddedUncertainty;
    info.RQCIMeanAddedScore = rqci_info.MeanAddedScore;
end
info.AfterUncertaintyCount = numel(candidx);

if isfield(cfg, 'enable_msm_representative_replace') && cfg.enable_msm_representative_replace
    [candidx, ecrs_info] = apply_msm_representative_replace_indices(pos, e, candidx, cfg, LB, UB);
    info.ECRSApplied = ecrs_info.Applied;
    info.ECRSRepresentativeRank = ecrs_info.RepresentativeRank;
    info.ECRSRepresentativenessGain = ecrs_info.RepresentativenessGain;
end
info.AfterRepresentativeCount = numel(candidx);

if cfg.enable_batch_decorrelation && numel(candidx) > 1
    info.PQDBDApplied = 1;
    [info.PQDBDMeanDistanceBefore, min_dist_before] = ...
        normalized_pairwise_distance_summary(pos, candidx, LB, UB);
    candidx = apply_batch_decorrelation_indices(pos, e, candidx, LB, UB, cfg);
    [info.PQDBDMeanDistanceAfter, min_dist_after] = ...
        normalized_pairwise_distance_summary(pos, candidx, LB, UB);
    info.PQDBDDiversityGain = info.PQDBDMeanDistanceAfter - info.PQDBDMeanDistanceBefore;
    info.PQDBDMinDistanceGain = min_dist_after - min_dist_before;
end

pos_trmem = pos(candidx, :);
info.FinalSelectedCount = numel(candidx);
if ~isempty(candidx)
    info.MeanSelectedUncertainty = mean(model_unc_all(candidx), 'omitnan');
end
end

function [explore_weight, keep_ratio] = effective_confidence_controls(cfg, err_ema)
explore_weight = cfg.acq_explore_weight;
keep_ratio = cfg.acq_keep_ratio;
if ~cfg.enable_confidence_allocation
    return;
end
signed_factor = (err_ema - cfg.confidence_error_target) / ...
    max(cfg.confidence_error_target, 1e-12);
signed_factor = max(-1.0, min(2.0, signed_factor));
explore_weight = clamp_scalar( ...
    cfg.acq_explore_weight * (1 + cfg.confidence_explore_gain * signed_factor), ...
    cfg.confidence_explore_min, cfg.confidence_explore_max);
keep_ratio = clamp_scalar( ...
    cfg.acq_keep_ratio * (1 + cfg.confidence_keep_gain * signed_factor), ...
    cfg.confidence_keep_min, cfg.confidence_keep_max);
end

function [candidx, info] = apply_msm_representative_replace_indices(pos, e, candidx, cfg, LB, UB)
info = struct('Applied', 0, 'RepresentativeRank', NaN, ...
    'RepresentativenessGain', NaN);
if isempty(candidx) || numel(candidx) <= 1
    return;
end

n = numel(candidx);
pool_n = round(cfg.msm_rep_pool_top_ratio * n);
pool_n = max(cfg.msm_rep_pool_top_min, pool_n);
pool_n = min(pool_n, cfg.msm_rep_pool_top_max);
pool_n = min(pool_n, n);
pool_n = max(pool_n, 2);
pool = candidx(1:pool_n);

r_low = max(1, min(cfg.msm_rep_random_top_min, pool_n));
r = randi([r_low, pool_n], 1, 1);
top_r = pool(1:r);
centroid = mean(pos(top_r, :), 1);
d = vecnorm(pos(pool, :) - centroid, 2, 2);
[~, rid] = min(d);
representative_idx = pool(rid);
keep_n = max(1, round(cfg.msm_rep_keep_ratio * n));
keep_n = min(keep_n, n);
[~, ord_e] = sort(e(candidx), 'ascend');
rank_all = candidx(ord_e);
representative_rank = find(rank_all == representative_idx, 1, 'first');
best_idx = rank_all(1);
domain_norm = max(norm(UB - LB), 1e-12);
representative_distance = norm(pos(representative_idx, :) - centroid) / domain_norm;
best_distance = norm(pos(best_idx, :) - centroid) / domain_norm;
if best_distance <= 1e-12
    representativeness_gain = 0;
else
    representativeness_gain = (best_distance - representative_distance) / best_distance;
end
rest = setdiff(rank_all, representative_idx, 'stable');
candidx = [representative_idx, rest(1:min(keep_n - 1, numel(rest)))];
info.Applied = 1;
info.RepresentativeRank = representative_rank;
info.RepresentativenessGain = representativeness_gain;
end

function [apply_ls, gap_scale] = get_dynamic_local_search_params(cfg, progress, stagnation_counter, gen)
apply_ls = true;
gap_scale = 1.0;
if ~isfield(cfg, 'enable_dynamic_trigger') || ~cfg.enable_dynamic_trigger
    return;
end

stag_th = max(1, round(cfg.dynamic_stagnation_threshold));
trigger_stag = stagnation_counter >= stag_th;
late_interval = max(1, round(cfg.dynamic_interval));
trigger_late = progress >= cfg.dynamic_late_progress && mod(gen, late_interval) == 0;
apply_ls = trigger_stag || trigger_late;
if ~apply_ls
    return;
end

stag_norm = min(1.0, stagnation_counter / stag_th);
gap_scale = 1 + cfg.dynamic_gap_stagnation_gain * stag_norm + cfg.dynamic_gap_progress_gain * progress;
gap_scale = clamp_scalar(gap_scale, cfg.dynamic_gap_min_scale, cfg.dynamic_gap_max_scale);
end

function [search_cfg_iter, gs_cfg_iter, use_ls_iter, stage_mode] = derive_iteration_policy( ...
    search_cfg_base, gs_cfg_base, use_ls_base, pos, LB, UB, progress, stagnation_counter, surrogate_err_ema)
search_cfg_iter = search_cfg_base;
gs_cfg_iter = gs_cfg_base;
use_ls_iter = use_ls_base;
stage_mode = 'balanced';
end

function cfg_out = apply_landscape_guard_gate(cfg_in, ghx, ghf, anchor, progress, stagnation_counter)
cfg_out = cfg_in;
if ~isfield(cfg_in, 'enable_landscape_guard_gate') || ~cfg_in.enable_landscape_guard_gate
    return;
end

corr_fd = helper_distance_fit_corr(ghx, ghf, anchor);
trigger_progress = progress >= cfg_in.landscape_gate_min_progress;
trigger_stag = stagnation_counter >= cfg_in.landscape_gate_min_stagnation;
is_rugged = corr_fd <= cfg_in.landscape_gate_rugged_corr_max;

if is_rugged && (trigger_progress || trigger_stag)
    return;
end

cfg_out.enable_confidence_allocation = false;
cfg_out.enable_surrogate_guard = false;
cfg_out.enable_uncertainty_infill = false;
if isfield(cfg_out, 'acq_explore_weight')
    cfg_out.acq_explore_weight = cfg_out.acq_explore_weight * cfg_in.landscape_gate_reduce_explore_weight;
end
if isfield(cfg_out, 'acq_keep_ratio')
    cfg_out.acq_keep_ratio = cfg_out.acq_keep_ratio * cfg_in.landscape_gate_reduce_keep_ratio;
end
cfg_out.acq_explore_weight = clamp_scalar(cfg_out.acq_explore_weight, 0.02, 0.65);
cfg_out.acq_keep_ratio = clamp_scalar(cfg_out.acq_keep_ratio, 0.45, 0.95);
end

function err_ema = update_surrogate_error_ema(err_ema, pred_vals, true_vals, cfg)
if ~(cfg.enable_confidence_allocation || (isfield(cfg, 'enable_surrogate_guard') && cfg.enable_surrogate_guard))
    return;
end
if isempty(pred_vals) || isempty(true_vals)
    return;
end

pred_vals = reshape(pred_vals, 1, []);
true_vals = reshape(true_vals, 1, []);
n = min(numel(pred_vals), numel(true_vals));
if n <= 0
    return;
end

pred_vals = pred_vals(1:n);
true_vals = true_vals(1:n);
scale = abs(true_vals) + cfg.confidence_err_scale_floor;
err = abs(true_vals - pred_vals) ./ scale;
err_mean = mean(err);
alpha = clamp_scalar(cfg.confidence_ema_alpha, 0.01, 1.0);
err_ema = (1 - alpha) * err_ema + alpha * err_mean;
end

function [dist_norm, dist_raw] = compute_min_distance_norm(pos, candidx, ref)
n = numel(candidx);
dist_raw = zeros(1, n);
if isempty(ref)
    dist_norm = zeros(1, n);
    return;
end

for ii = 1:n
    dists = vecnorm(ref - pos(candidx(ii), :), 2, 2);
    dist_raw(ii) = min(dists);
end
min_d = min(dist_raw);
max_d = max(dist_raw);
dist_norm = (dist_raw - min_d) ./ max(max_d - min_d, 1e-12);
end

function c = helper_distance_fit_corr(x, f, anchor)
c = 1.0;
if isempty(x) || isempty(f)
    return;
end

f = reshape(f, 1, []);
n = min(size(x,1), numel(f));
if n < 3
    return;
end

x = x(1:n, :);
f = f(1:n);
if nargin < 3 || isempty(anchor)
    [~, ib] = min(f);
    anchor = x(ib, :);
end

d = vecnorm(x - repmat(anchor, n, 1), 2, 2);
if std(d) <= 1e-12 || std(f) <= 1e-12
    c = 1.0;
    return;
end

C = corrcoef(d, f(:));
if numel(C) < 4 || ~isfinite(C(1,2))
    c = 1.0;
else
    c = C(1,2);
end
end

function sigma = build_sigma_from_uncertainty(unc_norm, hf, cfg)
if isempty(hf)
    f_scale = 1.0;
else
    f_scale = std(hf);
    if ~isfinite(f_scale) || f_scale <= 1e-12
        f_scale = max(mean(abs(hf)), 1.0);
    end
end

sigma = cfg.acq_sigma_floor + cfg.acq_sigma_scale .* unc_norm .* f_scale;
sigma = max(sigma, cfg.acq_sigma_floor);
end

function v_norm = normalize_vector(v)
v = reshape(v, 1, []);
if isempty(v)
    v_norm = v;
    return;
end
vmin = min(v);
vmax = max(v);
v_norm = (v - vmin) ./ max(vmax - vmin, 1e-12);
end

function [candidx, info] = apply_uncertainty_infill_indices(candidx, e, model_unc_all, cfg)
info = struct('AddedCount', 0, 'MeanAddedUncertainty', NaN, ...
    'MeanAddedScore', NaN);
n = numel(e);
if n <= 0
    return;
end

base_n = numel(candidx);
add_n = round(cfg.uncertainty_infill_ratio * max(base_n, 1));
add_n = max(cfg.uncertainty_infill_min, add_n);
add_n = min(cfg.uncertainty_infill_max, add_n);
if add_n <= 0
    return;
end

pool = setdiff(1:n, candidx, 'stable');
if isempty(pool)
    return;
end

unc_pool = normalize_vector(model_unc_all(pool));
e_pool = e(pool);
e_norm = normalize_vector(e_pool);
score = cfg.uncertainty_infill_model_weight .* unc_pool - (1 - cfg.uncertainty_infill_model_weight) .* e_norm;
[~, ord] = sort(score, 'descend');
pick_local = ord(1:min(add_n, numel(pool)));
pick = pool(pick_local);
candidx = unique([candidx, pick], 'stable');
info.AddedCount = numel(pick);
info.MeanAddedUncertainty = mean(model_unc_all(pick), 'omitnan');
info.MeanAddedScore = mean(score(pick_local), 'omitnan');
end

function [mean_distance, min_distance] = normalized_pairwise_distance_summary(pos, candidx, LB, UB)
% Summarize batch geometry in a unit-hypercube coordinate system so that
% PQDBD traces are comparable across dimensions and benchmark domains.
n = numel(candidx);
if n <= 1
    mean_distance = NaN;
    min_distance = NaN;
    return;
end
domain = max(UB - LB, 1e-12);
X = (pos(candidx, :) - repmat(LB, n, 1)) ./ repmat(domain, n, 1);
distance_count = n * (n - 1) / 2;
distances = zeros(distance_count, 1);
cursor = 1;
for i = 1:(n - 1)
    d = vecnorm(X((i + 1):n, :) - X(i, :), 2, 2) ./ ...
        sqrt(max(1, size(X, 2)));
    next_cursor = cursor + numel(d) - 1;
    distances(cursor:next_cursor) = d;
    cursor = next_cursor + 1;
end
mean_distance = mean(distances, 'omitnan');
min_distance = min(distances);
end

function vals = evaluate_exact_batch(fname, x_batch, cfg)
n = size(x_batch, 1);
vals = zeros(1, n);
if n == 0
    return;
end

use_parallel = cfg.enable_parallel_exact_eval ...
    && n >= cfg.parallel_min_batch ...
    && is_parallel_pool_ready();
if ~use_parallel
    for i = 1:n
        vals(i) = feval(fname, x_batch(i,:));
    end
    return;
end

vals_col = zeros(n,1);
parfor i = 1:n
    vals_col(i) = feval(fname, x_batch(i,:));
end
vals = vals_col.';
end

function tf = is_parallel_pool_ready()
tf = false;
if isempty(ver('parallel'))
    return;
end
try
    p = gcp('nocreate');
    tf = ~isempty(p);
catch
    tf = false;
end
end

function candidx = apply_batch_decorrelation_indices(pos, e, candidx, LB, UB, cfg)
if numel(candidx) <= 1
    return;
end

if isfield(cfg, 'batch_diversity_mode')
    target_n = max(1, round(cfg.batch_min_keep_ratio * numel(candidx)));
    if strcmpi(cfg.batch_diversity_mode, 'maxmin')
        candidx = apply_batch_maxmin_indices(pos, candidx, target_n);
        return;
    elseif strcmpi(cfg.batch_diversity_mode, 'kmeanspp')
        candidx = apply_batch_kmeanspp_indices(pos, e, candidx, target_n, LB, UB, cfg);
        return;
    end
end

D = size(pos, 2);
domain = UB - LB;
min_dist = cfg.batch_min_dist_ratio * norm(domain) / sqrt(max(1, D));
if min_dist <= 0
    return;
end

selected = zeros(1, numel(candidx));
selected_count = 0;
for ii = 1:numel(candidx)
    idx = candidx(ii);
    if selected_count == 0
        selected_count = 1;
        selected(selected_count) = idx;
        continue;
    end
    prev_idx = selected(1:selected_count);
    dists = vecnorm(pos(prev_idx, :) - pos(idx, :), 2, 2);
    if min(dists) >= min_dist
        selected_count = selected_count + 1;
        selected(selected_count) = idx;
    end
end

min_keep = max(1, round(cfg.batch_min_keep_ratio * numel(candidx)));
selected = selected(1:selected_count);
if selected_count < min_keep
    remain = setdiff(candidx, selected, 'stable');
    need = min_keep - selected_count;
    selected = [selected, remain(1:min(need, numel(remain)))];
end

candidx = selected;
end

function candidx = apply_batch_maxmin_indices(pos, candidx, target_n)
n = numel(candidx);
target_n = min(max(1, target_n), n);
if target_n >= n
    return;
end

selected = zeros(1, target_n);
selected(1) = candidx(1); % preserve the top-ranked candidate

for k = 2:target_n
    remain = setdiff(candidx, selected(1:k-1), 'stable');
    best_idx = remain(1);
    best_val = -inf;
    for ii = 1:numel(remain)
        ridx = remain(ii);
        d = vecnorm(pos(selected(1:k-1), :) - pos(ridx, :), 2, 2);
        dmin = min(d);
        if dmin > best_val
            best_val = dmin;
            best_idx = ridx;
        end
    end
    selected(k) = best_idx;
end

candidx = selected;
end

function candidx = apply_batch_kmeanspp_indices(pos, e, candidx, target_n, LB, UB, cfg)
n = numel(candidx);
target_n = min(max(1, target_n), n);
if target_n >= n
    return;
end

e = reshape(e, 1, []);
X = pos(candidx, :);
domain = max(UB - LB, 1e-12);
X = (X - repmat(LB, n, 1)) ./ repmat(domain, n, 1);
mu = mean(X, 1);
sd = std(X, 0, 1);
sd = max(sd, 1e-6);
X = (X - repmat(mu, n, 1)) ./ repmat(sd, n, 1);

proj_dim = 24;
if isfield(cfg, 'batch_kmeanspp_proj_dim')
    proj_dim = max(2, round(cfg.batch_kmeanspp_proj_dim));
end
if size(X, 2) > proj_dim
    Xc = X - repmat(mean(X, 1), n, 1);
    [~, S, V] = svd(Xc, 'econ');
    sing_vals = diag(S);
    if ~isempty(sing_vals)
        keep = min(proj_dim, numel(sing_vals));
        if sing_vals(1) > 1e-12
            rel = sing_vals ./ sing_vals(1);
            keep = min(keep, max(2, sum(rel > 1e-3)));
        end
        X = Xc * V(:, 1:keep);
    end
end

quality_bias = 0.35;
if isfield(cfg, 'batch_kmeanspp_quality_bias')
    quality_bias = max(0, cfg.batch_kmeanspp_quality_bias);
end
rank_norm = (0:(n-1)) ./ max(1, (n-1));
quality_weight = 1 + quality_bias .* (1 - rank_norm);
greedy_pick = true;
if isfield(cfg, 'batch_kmeanspp_greedy_pick')
    greedy_pick = logical(cfg.batch_kmeanspp_greedy_pick);
end

selected_local = zeros(1, target_n);
selected_local(1) = 1; % keep best-ranked candidate (already sorted by e)
selected_count = 1;

for k = 2:target_n
    remain = setdiff(1:n, selected_local(1:selected_count), 'stable');
    if isempty(remain)
        break;
    end

    d2 = zeros(1, numel(remain));
    for ii = 1:numel(remain)
        ridx_local = remain(ii);
        d = vecnorm(X(selected_local(1:selected_count), :) - X(ridx_local, :), 2, 2);
        dmin = min(d);
        d2(ii) = (dmin * dmin) * quality_weight(ridx_local);
    end

    s = sum(d2);
    if s <= 1e-12
        [~, pick_local] = min(e(candidx(remain)));
        pick = remain(pick_local);
    elseif greedy_pick
        [~, pick_local] = max(d2);
        pick = remain(pick_local);
    else
        p = d2 ./ s;
        cdf = cumsum(p);
        r = rand;
        jj = find(cdf >= r, 1, 'first');
        if isempty(jj)
            jj = numel(remain);
        end
        pick = remain(jj);
    end
    selected_count = selected_count + 1;
    selected_local(selected_count) = pick;
end

selected_local = selected_local(selected_local > 0);
refine_iters = 2;
if isfield(cfg, 'batch_kmeanspp_refine_iters')
    refine_iters = max(0, round(cfg.batch_kmeanspp_refine_iters));
end
if refine_iters > 0 && numel(selected_local) > 1
    centers = X(selected_local, :);
    rank_tradeoff = 0.20;
    if isfield(cfg, 'batch_kmeanspp_rank_tradeoff')
        rank_tradeoff = max(0, cfg.batch_kmeanspp_rank_tradeoff);
    end
    for it = 1:refine_iters
        lab = assign_points_to_centers(X, centers);
        for c = 1:size(centers,1)
            idx_c = find(lab == c);
            if ~isempty(idx_c)
                centers(c, :) = mean(X(idx_c, :), 1);
            end
        end
    end

    selected_refined = zeros(1, size(centers,1));
    used_local = false(1, n);
    for c = 1:size(centers,1)
        d = vecnorm(X - centers(c, :), 2, 2);
        d(used_local) = inf;
        if all(~isfinite(d))
            remain = find(~used_local);
            [~, rid] = min(e(candidx(remain)));
            id_pick = remain(rid);
        else
            d_norm = d ./ max(max(d(isfinite(d))), 1e-12);
            score = d_norm + rank_tradeoff .* rank_norm(:);
            score(~isfinite(score)) = inf;
            [~, id_pick] = min(score);
        end
        selected_refined(c) = id_pick;
        used_local(id_pick) = true;
    end
    selected_local = unique(selected_refined, 'stable');
end

if numel(selected_local) < target_n
    remain = setdiff(1:n, selected_local, 'stable');
    need = target_n - numel(selected_local);
    selected_local = [selected_local, remain(1:min(need, numel(remain)))];
end

selected_global = candidx(selected_local);
[~, ord_best] = sort(e(selected_global), 'ascend');
candidx = selected_global(ord_best);
end

function lab = assign_points_to_centers(X, centers)
n = size(X, 1);
k = size(centers, 1);
dist_mat = zeros(n, k);
for c = 1:k
    diff = X - centers(c, :);
    dist_mat(:, c) = sum(diff .* diff, 2);
end
[~, lab] = min(dist_mat, [], 2);
end

function y = clamp_scalar(x, lo, hi)
y = min(max(x, lo), hi);
end

function FUN = wrap_surrogate_predictor(FUN_raw, calib_scale, calib_bias, cfg)
if ~cfg.enable_surrogate_calibration
    FUN = FUN_raw;
    return;
end
FUN = @(x) apply_surrogate_calibration_values(FUN_raw(x), calib_scale, calib_bias);
end

function [scale_new, bias_new] = update_surrogate_calibration(scale_old, bias_old, pred_vals, true_vals, hf_ref, cfg)
scale_new = scale_old;
bias_new = bias_old;
if ~cfg.enable_surrogate_calibration
    return;
end
if isempty(pred_vals) || isempty(true_vals)
    return;
end

pred_vals = pred_vals(:);
true_vals = true_vals(:);
n = min(numel(pred_vals), numel(true_vals));
if n <= 0
    return;
end
pred_vals = pred_vals(1:n);
true_vals = true_vals(1:n);

if n >= 2 && std(pred_vals) > cfg.calib_eps
    X = [pred_vals, ones(n,1)];
    theta = X \ true_vals;
    scale_batch = theta(1);
    bias_batch = theta(2);
else
    scale_batch = 1.0;
    bias_batch = mean(true_vals - pred_vals);
end

if ~isfinite(scale_batch)
    scale_batch = 1.0;
end
if ~isfinite(bias_batch)
    bias_batch = 0.0;
end

if isempty(hf_ref)
    hf_scale = max(cfg.calib_eps, mean(abs(true_vals)));
else
    hf_scale = max(cfg.calib_eps, std(hf_ref));
end

scale_target = clamp_scalar(scale_batch, cfg.calib_scale_min, cfg.calib_scale_max);
bias_clip = cfg.calib_bias_clip_ratio * hf_scale;
bias_target = clamp_scalar(bias_batch, -bias_clip, bias_clip);
alpha = clamp_scalar(cfg.calib_alpha, 0.01, 1.0);

scale_new = (1 - alpha) * scale_old + alpha * scale_target;
bias_new = (1 - alpha) * bias_old + alpha * bias_target;
end

function cfg = merge_cfg(default_cfg, user_cfg)
cfg = default_cfg;
if isempty(user_cfg)
    return;
end

fn = fieldnames(user_cfg);
for i = 1:numel(fn)
    cfg.(fn{i}) = user_cfg.(fn{i});
end
end

function [ghx, ghf] = build_training_subset(hx, hf, gs, LB, UB, cfg, anchor_x)
[hf_sorted, id_sorted] = sort(hf);
gs_eff = min(gs, numel(hf_sorted));

sel_idx = id_sorted(1:gs_eff);
ghx = hx(sel_idx, :);
ghf = hf(sel_idx);
end

function [hx_new, hf_new, is_added] = upsert_history_point(hx, hf, x_new, f_new, LB, UB, cfg)
hx_new = hx;
hf_new = hf;
is_added = false;

if isempty(hx_new)
    hx_new = x_new;
    hf_new = f_new;
    is_added = true;
    return;
end

if ~cfg.enable_history_dedup
    hx_new = [hx_new; x_new];
    hf_new = [hf_new, f_new];
    is_added = true;
    return;
end

tol = cfg.history_dedup_tol_ratio * norm(UB - LB) / sqrt(size(hx_new,2));
d = vecnorm(hx_new - x_new, 2, 2);
[dmin, idx] = min(d);

if dmin <= tol
    if f_new < hf_new(idx)
        hx_new(idx, :) = x_new;
        hf_new(idx) = f_new;
    end
else
    hx_new = [hx_new; x_new];
    hf_new = [hf_new, f_new];
    is_added = true;
end
end

function model = build_surrogate_predictor(ghx, ghf, D, cfg, LB, UB)
n = size(ghx,1);
model = struct();
model.cfg = cfg;
model.use_constant = false;
model.constant_value = mean(ghf);
model.f_scale = std(ghf);
if ~isfinite(model.f_scale) || model.f_scale <= 1e-12
    model.f_scale = max(mean(abs(ghf)), 1.0);
end
model.net_global = [];
model.global_norm_ctx = struct('enabled', false);
model.global_x_norm = [];
model.global_repr_state = struct('enabled', false);
model.local_models = struct('net', {}, 'norm_ctx', {}, 'x_norm', {}, 'center_global', {});
model.local_centers = zeros(0, D);

% Initialize GPR model fields
model.gpr_model = [];
model.hybrid_model = [];

if n <= 1
    model.use_constant = true;
    return;
end

[x_global, norm_ctx] = normalize_input_points(ghx, LB, UB, cfg);
[x_global_repr, repr_state] = build_representation_stack_train(x_global, ghf, cfg);
model.net_global = build_rbf_net(x_global_repr, ghf, D, cfg.ensemble_global_spr_scale, cfg);
model.global_norm_ctx = norm_ctx;
model.global_x_norm = x_global_repr;
model.global_repr_state = repr_state;

if cfg.enable_hierarchical_surrogate
    model.local_models = build_local_surrogate_models(ghx, ghf, x_global, D, cfg, LB, UB);
elseif cfg.enable_surrogate_ensemble && n >= max(4, cfg.ensemble_local_min_points)
    [ghf_sorted, idx_sorted] = sort(ghf);
    local_n = min(n, max(cfg.ensemble_local_min_points, round(cfg.ensemble_local_ratio * n)));
    local_idx = idx_sorted(1:local_n);
    local_x = ghx(local_idx, :);
    local_f = ghf_sorted(1:local_n);
    [x_local, local_norm_ctx] = normalize_input_points(local_x, LB, UB, cfg);
    local_model = struct();
    local_model.net = build_rbf_net(x_local, local_f, D, cfg.ensemble_local_spr_scale, cfg);
    local_model.norm_ctx = local_norm_ctx;
    local_model.x_norm = x_local;
    local_model.center_global = mean(x_global(local_idx, :), 1);
    model.local_models = local_model;
end

if ~isempty(model.local_models)
    model.local_centers = reshape([model.local_models.center_global], size(x_global,2), []).';
end

% Build hybrid GPR-RBF model if enabled
if isfield(cfg, 'enable_hybrid_gpr_rbf') && cfg.enable_hybrid_gpr_rbf && n >= cfg.gpr_min_samples
    try
        % Build hybrid model using GPR for uncertainty and RBF for prediction
        model.hybrid_model = build_hybrid_surrogate_model(ghx, ghf, D, cfg, LB, UB, ...
            'enable_gpr', true, ...
            'enable_rbf_ensemble', false, ...  % Use existing RBF
            'optimize_gpr', cfg.gpr_enable_optimize);
    catch ME
        warning('Failed to build hybrid GPR model: %s', ME.message);
        model.hybrid_model = [];
    end
end
end

function y = surrogate_predict_mean(model, x)
if isempty(x)
    y = zeros(1,0);
    return;
end
if size(x,1) == 1 && isvector(x)
    x = reshape(x, 1, []);
end

nq = size(x,1);
if model.use_constant
    y = model.constant_value * ones(1, nq);
    return;
end

cfg = model.cfg;
x_global = normalize_query_points(x, model.global_norm_ctx, cfg);
x_global_repr = apply_representation_stack_query(x_global, model.global_repr_state, cfg);
y_global = sim(model.net_global, x_global_repr').';
if isempty(model.local_models) || ~isfield(cfg, 'hier_use_local_for_mean') || ~cfg.hier_use_local_for_mean
    y = y_global;
    return;
end

y_local = y_global;
for i = 1:nq
    d_centers = vecnorm(model.local_centers - x_global(i,:), 2, 2);
    [~, cid] = min(d_centers);
    lm = model.local_models(cid);
    x_loc = normalize_query_points(x(i,:), lm.norm_ctx, cfg);
    y_local(i) = sim(lm.net, x_loc');
end

if isfield(cfg, 'hier_collab_mode') && strcmpi(cfg.hier_collab_mode, 'weighted')
    wg = cfg.ensemble_global_weight;
    wl = cfg.ensemble_local_weight;
    wsum = max(wg + wl, 1e-12);
    wg = wg / wsum;
    wl = wl / wsum;
    y = wg .* y_global + wl .* y_local;
else
    y = 0.5 .* y_global + 0.5 .* y_local;
end
end

function u = surrogate_predict_uncertainty(model, x)
[~, u] = surrogate_predict_mean_uncertainty(model, x);
end

function [y, u] = surrogate_predict_mean_uncertainty(model, x)
if isempty(x)
    y = zeros(1,0);
    u = zeros(1,0);
    return;
end
if size(x,1) == 1 && isvector(x)
    x = reshape(x, 1, []);
end

nq = size(x,1);
if model.use_constant
    y = model.constant_value * ones(1, nq);
    u = ones(1, nq);
    return;
end

% Use hybrid GPR-RBF model if available for analytical uncertainty
if isfield(model, 'hybrid_model') && ~isempty(model.hybrid_model)
    try
        [y_gpr, sigma2_gpr] = hybrid_predict(model.hybrid_model, x);
        % Convert variance to normalized uncertainty
        % Use RBF prediction for mean, GPR variance for uncertainty
        cfg = model.cfg;
        x_global = normalize_query_points(x, model.global_norm_ctx, cfg);
        x_global_repr = apply_representation_stack_query(x_global, model.global_repr_state, cfg);
        y_rbf = sim(model.net_global, x_global_repr')';
        y_rbf = reshape(y_rbf, 1, []);
        
        % Normalize GPR variance to [0, 1] range
        sigma2_gpr = reshape(sigma2_gpr, 1, []);
        f_scale = max(model.f_scale, 1e-12);
        u_gpr_normalized = sqrt(sigma2_gpr) / f_scale;
        u_gpr_normalized = max(0, min(2, u_gpr_normalized));
        
        % Return RBF prediction with GPR-based uncertainty
        y = y_rbf;
        u = u_gpr_normalized;
        return;
    catch ME
        warning('Hybrid model prediction failed, falling back to original method: %s', ME.message);
    end
end

cfg = model.cfg;
x_global = normalize_query_points(x, model.global_norm_ctx, cfg);
x_global_repr = apply_representation_stack_query(x_global, model.global_repr_state, cfg);
y_global = sim(model.net_global, x_global_repr').';
u_global = min_distance_to_reference(x_global_repr, model.global_x_norm).' ./ sqrt(max(1, size(x_global_repr,2)));
u_global = max(0, min(1, u_global));

if isempty(model.local_models)
    y = y_global;
    u = u_global;
    return;
end

y_local = y_global;
u_local = inf(1, nq);
local_ok = false(1, nq);
for i = 1:nq
    d_centers = vecnorm(model.local_centers - x_global(i,:), 2, 2);
    [~, cid] = min(d_centers);
    lm = model.local_models(cid);
    x_loc = normalize_query_points(x(i,:), lm.norm_ctx, cfg);
    y_local(i) = sim(lm.net, x_loc');
    d_loc = min_distance_to_reference(x_loc, lm.x_norm);
    u_local(i) = d_loc ./ sqrt(max(1, size(x_loc,2)));
    local_ok(i) = true;
end
u_local = max(0, min(1, u_local));

if isfield(cfg, 'hier_collab_mode') && strcmpi(cfg.hier_collab_mode, 'weighted')
    wg = 1 ./ max(u_global, 1e-6);
    wl = 1 ./ max(u_local, 1e-6);
    y = (wg .* y_global + wl .* y_local) ./ max(wg + wl, 1e-12);
    u_base = (u_global .* u_local) ./ max(u_global + u_local, 1e-12);
else
    y = y_global;
    use_local = local_ok & (u_local < u_global);
    y(use_local) = y_local(use_local);
    u_base = min(u_global, u_local);
end

disagree = abs(y_global - y_local) ./ max(model.f_scale, 1e-12);
disagree = min(disagree, 2.0);
u = u_base + cfg.hier_uncertainty_alpha .* disagree;
u(~local_ok) = u_global(~local_ok);
u = max(0, min(2, u));
end

function [labels, centers] = kmeanspp_cluster_points(x, k, max_iter)
n = size(x,1);
d = size(x,2);
k = min(max(1, round(k)), n);
max_iter = max(1, round(max_iter));

centers = zeros(k, d);
first_idx = randi(n);
centers(1,:) = x(first_idx,:);
for c = 2:k
    d2 = zeros(n,1);
    for i = 1:n
        dd = vecnorm(centers(1:c-1,:) - x(i,:), 2, 2);
        d2(i) = min(dd).^2;
    end
    s = sum(d2);
    if s <= 1e-12
        pick = randi(n);
    else
        cdf = cumsum(d2 ./ s);
        r = rand;
        pick = find(cdf >= r, 1, 'first');
        if isempty(pick)
            pick = n;
        end
    end
    centers(c,:) = x(pick,:);
end

labels = ones(n,1);
for it = 1:max_iter
    changed = false;
    for i = 1:n
        dists = vecnorm(centers - x(i,:), 2, 2);
        [~, best_k] = min(dists);
        if labels(i) ~= best_k
            labels(i) = best_k;
            changed = true;
        end
    end

    for c = 1:k
        idx = find(labels == c);
        if isempty(idx)
            all_min = min_distance_to_reference(x, centers);
            [~, far_idx] = max(all_min);
            centers(c,:) = x(far_idx,:);
            labels(far_idx) = c;
        else
            centers(c,:) = mean(x(idx,:), 1);
        end
    end

    if ~changed
        break;
    end
end
end

function dmin = min_distance_to_reference(xq, xref)
nq = size(xq,1);
dmin = zeros(nq,1);
if isempty(xref)
    dmin(:) = 1.0;
    return;
end
for i = 1:nq
    d = vecnorm(xref - xq(i,:), 2, 2);
    dmin(i) = min(d);
end
end

function [x_norm, ctx] = normalize_input_points(x, LB, UB, cfg)
ctx = struct();
if ~cfg.enable_input_normalization
    x_norm = x;
    ctx.enabled = false;
    return;
end

LB = reshape(LB, 1, []);
UB = reshape(UB, 1, []);
if numel(LB) ~= size(x,2) || numel(UB) ~= size(x,2)
    lb_use = min(x, [], 1);
    ub_use = max(x, [], 1);
else
    lb_use = LB;
    ub_use = UB;
end

scale = ub_use - lb_use;
scale = max(scale, 1e-12);
x_norm = (x - repmat(lb_use, size(x,1), 1)) ./ repmat(scale, size(x,1), 1);

ctx.enabled = true;
ctx.lb = lb_use;
ctx.scale = scale;
end

function x_norm = normalize_query_points(x, ctx, cfg)
if ~ctx.enabled
    x_norm = x;
    return;
end

x_norm = (x - repmat(ctx.lb, size(x,1), 1)) ./ repmat(ctx.scale, size(x,1), 1);
if isfield(cfg, 'input_norm_clip') && cfg.input_norm_clip
    x_norm = min(max(x_norm, 0), 1);
end
end

function local_models = build_local_surrogate_models(ghx, ghf, x_global, D, cfg, LB, UB)
local_models = struct('net', {}, 'norm_ctx', {}, 'x_norm', {}, 'center_global', {});
n = size(ghx, 1);
min_points = max(4, round(cfg.hier_local_min_points));
k = min(cfg.hier_num_local_models, floor(n / min_points));
if k < 2
    return;
end

[labels, centers] = kmeanspp_cluster_points(x_global, k, cfg.hier_kmeans_iters);
for cid = 1:k
    idx = find(labels == cid);
    if numel(idx) < min_points
        continue;
    end
    local_x = ghx(idx, :);
    local_f = ghf(idx);
    [x_local, local_norm_ctx] = normalize_input_points(local_x, LB, UB, cfg);
    lm = struct();
    lm.net = build_rbf_net(x_local, local_f, D, cfg.ensemble_local_spr_scale, cfg);
    lm.norm_ctx = local_norm_ctx;
    lm.x_norm = x_local;
    lm.center_global = centers(cid, :);
    local_models(end+1) = lm; %#ok<AGROW>
end
end

function [x_repr, state] = build_representation_stack_train(x, y, cfg)
state = struct();
state.enabled = false;
state.attn_gain = ones(1, size(x,2));
state.inr = struct('enabled', false);
state.contrastive_gain = ones(1, size(x,2));
state.diffusion_A = eye(size(x,2));
state.feature_scale = ones(1, size(x,2));

if isempty(x) || ~(isfield(cfg, 'enable_representation_stack') && cfg.enable_representation_stack)
    x_repr = x;
    return;
end

x_repr = x;
end

function x_repr = apply_representation_stack_query(x, state, cfg)
if isempty(x)
    x_repr = x;
    return;
end
if isempty(state) || ~isfield(state, 'enabled') || ~state.enabled
    x_repr = x;
    return;
end
x_repr = x;
end

function net = build_rbf_net(x, y, D, spr_scale, cfg)
if nargin < 5 || isempty(cfg)
    cfg = struct();
end
y = reshape(y, 1, []);
[x_use, y_use] = prepare_rbf_training_samples(x, y, cfg, 1.0);
if isfield(cfg, 'enable_rbf_train_stabilizer') && cfg.enable_rbf_train_stabilizer
    spr_try = compute_rbf_spread(x_use, D, spr_scale, cfg);
    rc = estimate_rbf_kernel_rcond(x_use, spr_try);
    if rc < cfg.rbf_stab_rcond_threshold && size(x_use,1) > max(4, cfg.rbf_stab_min_points)
        [x_retry, y_retry] = prepare_rbf_training_samples(x_use, y_use, cfg, cfg.rbf_stab_retry_dist_scale);
        if size(x_retry,1) >= max(4, cfg.rbf_stab_min_points) && size(x_retry,1) <= size(x_use,1)
            x_use = x_retry;
            y_use = y_retry;
            spr_scale = spr_scale * cfg.rbf_stab_retry_spr_scale;
        end
    end
end

n = size(x_use,1);
if n <= 1
    net = newrbe(x_use', y_use, 1.0);
    return;
end
spr = compute_rbf_spread(x_use, D, spr_scale, cfg);
net = newrbe(x_use', y_use, spr);
end

function [x_keep, y_keep] = prepare_rbf_training_samples(x, y, cfg, dist_scale)
y = reshape(y, 1, []);
n = size(x, 1);
if n <= 1
    x_keep = x;
    y_keep = y;
    return;
end

[y_sort, ord] = sort(y, 'ascend');
x_sort = x(ord, :);
[~, ia] = unique(x_sort, 'rows', 'stable');
x_u = x_sort(ia, :);
y_u = y_sort(ia);

if ~(isfield(cfg, 'enable_rbf_train_stabilizer') && cfg.enable_rbf_train_stabilizer)
    x_keep = x_u;
    y_keep = y_u;
    return;
end

min_points = max(4, round(cfg.rbf_stab_min_points));
max_points = max(min_points, round(cfg.rbf_stab_max_points));
if size(x_u, 1) > max_points
    [x_u, y_u] = truncate_rbf_samples_by_quality_diversity(x_u, y_u, max_points, cfg);
end

D = size(x_u, 2);
domain_norm = norm(max(x_u, [], 1) - min(x_u, [], 1));
if ~isfinite(domain_norm) || domain_norm <= 1e-12
    domain_norm = sqrt(max(1, D));
end
if nargin < 4 || isempty(dist_scale)
    dist_scale = 1.0;
end
min_dist = cfg.rbf_stab_min_dist_ratio * dist_scale * domain_norm / sqrt(max(1, D));
min_dist = max(min_dist, 0.0);

keep_idx = zeros(1, size(x_u, 1));
keep_n = 1;
keep_idx(1) = 1;
for i = 2:size(x_u, 1)
    d = vecnorm(x_u(keep_idx(1:keep_n), :) - x_u(i, :), 2, 2);
    if isempty(d) || min(d) >= min_dist
        keep_n = keep_n + 1;
        keep_idx(keep_n) = i;
    end
end
keep_idx = keep_idx(1:keep_n);

if numel(keep_idx) < min_points
    remain = setdiff(1:size(x_u,1), keep_idx, 'stable');
    need = min_points - numel(keep_idx);
    keep_idx = [keep_idx, remain(1:min(need, numel(remain)))];
end

x_keep = x_u(keep_idx, :);
y_keep = y_u(keep_idx);
end

function [x_sel, y_sel] = truncate_rbf_samples_by_quality_diversity(x, y, max_points, cfg)
n = size(x, 1);
if n <= max_points
    x_sel = x;
    y_sel = y;
    return;
end

elite_n = max(1, min(max_points - 1, round(cfg.rbf_stab_elite_ratio * max_points)));
sel = zeros(1, max_points);
sel_n = elite_n;
sel(1:elite_n) = 1:elite_n;

while sel_n < max_points
    remain = setdiff(1:n, sel(1:sel_n), 'stable');
    if isempty(remain)
        break;
    end
    d_best = -inf;
    pick = remain(1);
    for i = 1:numel(remain)
        rid = remain(i);
        d = vecnorm(x(sel(1:sel_n), :) - x(rid, :), 2, 2);
        dmin = min(d);
        if dmin > d_best
            d_best = dmin;
            pick = rid;
        end
    end
    sel_n = sel_n + 1;
    sel(sel_n) = pick;
end
sel = sel(1:sel_n);
x_sel = x(sel, :);
y_sel = y(sel);
end

function spr = compute_rbf_spread(x, D, spr_scale, cfg)
n = size(x, 1);
if n <= 1
    spr = 1.0;
    return;
end
xd = real(sqrt(max(0, x.^2*ones(size(x')) + ones(size(x))*(x').^2 - 2*x*(x'))));
xd_pos = xd(xd > 1e-12);
if isempty(xd_pos)
    spr_base = 1.0;
    med_d = 1.0;
else
    spr_base = max(xd_pos) / (D * max(1, n))^(1 / max(1, D));
    med_d = median(xd_pos);
end
spr = max(1e-12, spr_base * spr_scale);
if isfield(cfg, 'enable_rbf_train_stabilizer') && cfg.enable_rbf_train_stabilizer
    low = max(1e-12, cfg.rbf_stab_spread_min_factor * med_d);
    high = max(low, cfg.rbf_stab_spread_max_factor * med_d);
    spr = clamp_scalar(spr, low, high);
end
end

function rc = estimate_rbf_kernel_rcond(x, spr)
n = size(x, 1);
if n <= 2
    rc = 1.0;
    return;
end
xd = real(sqrt(max(0, x.^2*ones(size(x')) + ones(size(x))*(x').^2 - 2*x*(x'))));
phi = exp(-(xd.^2) ./ max(2 * spr * spr, 1e-24));
phi = phi + 1e-12 * eye(n);
rc = rcond(phi);
if ~isfinite(rc)
    rc = 0.0;
end
end

%% ============================================================================
% HYBRID GPR-RBF SURROGATE MODEL
% Integrates GPR with Matern 3/2 kernel for analytical uncertainty quantification
% Combined with RBF for fast and accurate prediction
% Reference: Ma et al., "A surrogate-assisted evolutionary algorithm with Gaussian
%            process", Applied Soft Computing 182 (2025) 113440
% ============================================================================

function model = build_hybrid_surrogate_model(ghx, ghf, D, cfg, LB, UB, varargin)
% BUILD_HYBRID_SURROGATE_MODEL - Build hybrid GPR-RBF surrogate
%
% This function builds a hybrid surrogate model that combines:
%   - RBF network for fast and accurate prediction
%   - GPR with Matern 3/2 kernel for analytical uncertainty quantification
%
% Inputs:
%   ghx - Historical X data (n x D)
%   ghf - Historical function values (n x 1)
%   D   - Problem dimension
%   cfg - Configuration struct
%   LB, UB - Variable bounds
%
% Optional params:
%   'enable_gpr' - Enable GPR uncertainty (default: true)
%   'enable_rbf_ensemble' - Enable RBF multi-kernel ensemble (default: true)
%   'optimize_gpr' - Optimize GPR hyperparameters (default: true)
%
% Output:
%   model - Hybrid surrogate model struct with fields:
%     .rbf_model - RBF model for prediction
%     .gpr_model - GPR model for uncertainty (or empty if disabled)
%     .enable_gpr - Whether GPR is enabled
%     .enable_rbf_ensemble - Whether RBF ensemble is enabled
%     .f_scale - Function value scale for normalization
%     .norm_ctx - Normalization context for GPR

    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'enable_gpr', true);
    addParameter(p, 'enable_rbf_ensemble', true);
    addParameter(p, 'optimize_gpr', true);
    parse(p, varargin{:});
    opts = p.Results;

    n = size(ghx, 1);
    model = struct();
    model.cfg = cfg;
    model.enable_gpr = opts.enable_gpr;
    model.enable_rbf_ensemble = opts.enable_rbf_ensemble;
    model.rbf_model = [];
    model.gpr_model = [];
    model.f_scale = std(ghf);
    if ~isfinite(model.f_scale) || model.f_scale <= 1e-12
        model.f_scale = max(mean(abs(ghf)), 1.0);
    end

    % Normalization for GPR
    [x_norm, norm_ctx] = normalize_input_points(ghx, LB, UB, cfg);
    model.norm_ctx = norm_ctx;
    model.LB = LB;
    model.UB = UB;

    % Build RBF model for mean prediction
    if opts.enable_rbf_ensemble && isfield(cfg, 'enable_rbf_ensemble') && cfg.enable_rbf_ensemble
        model.rbf_model = build_rbf_ensemble(x_norm, ghf, cfg);
        model.rbf_model_type = 'triple_ensemble';
    else
        model.rbf_model = build_rbf_net(x_norm, ghf, D, cfg.ensemble_global_spr_scale, cfg);
        model.rbf_model_type = 'single_mq';
    end

    % Build GPR model for uncertainty quantification
    if opts.enable_gpr && n >= 5
        try
            model.gpr_model = gpr_train_hybrid(x_norm, ghf, 'optimize', opts.optimize_gpr);
        catch
            model.gpr_model = [];
        end
    end
end


function [y, sigma2] = hybrid_predict(model, x)
% HYBRID_PREDICT - Predict mean and uncertainty using hybrid model
%
% Inputs:
%   model - Hybrid surrogate model from build_hybrid_surrogate_model
%   x     - Query points (nq x D)
%
% Outputs:
%   y     - Predicted mean values (nq x 1)
%   sigma2 - Predicted variance (nq x 1)

    if isempty(x)
        y = zeros(1, 0);
        sigma2 = zeros(1, 0);
        return;
    end

    if size(x, 1) == 1 && isvector(x)
        x = reshape(x, 1, []);
    end

    nq = size(x, 1);

    % Normalize query points
    x_norm = normalize_query_points(x, model.norm_ctx, model.cfg);

    % RBF prediction for mean
    if ~isempty(model.rbf_model)
        if isfield(model, 'rbf_model_type') && strcmp(model.rbf_model_type, 'triple_ensemble')
            % Triple-kernel RBF ensemble prediction
            y = rbf_ensemble_predict(model.rbf_model, x_norm);
            y = reshape(y, nq, 1);
        else
            % Original single-kernel RBF
            y = sim(model.rbf_model, x_norm')';
            y = reshape(y, nq, 1);
        end
    else
        y = zeros(nq, 1);  % Fallback to zero
    end

    % GPR prediction for uncertainty (analytical variance)
    if ~isempty(model.gpr_model)
        try
            [mu_gpr, var_gpr] = gpr_predict_hybrid(model.gpr_model, x_norm);
            sigma2 = reshape(var_gpr, nq, 1);
            % Combine RBF mean with GPR uncertainty
            % Weight GPR uncertainty more for far-from-training points
            y = y;  % Use RBF prediction as primary mean
        catch
            % Fallback to distance-based uncertainty
            sigma2 = compute_distance_uncertainty(x_norm, model.norm_ctx);
        end
    else
        % Fallback to distance-based uncertainty
        sigma2 = compute_distance_uncertainty(x_norm, model.norm_ctx);
    end

    % Ensure positive variance
    sigma2 = max(sigma2, 1e-12);
end


function model = gpr_train_hybrid(X, y, varargin)
% GPR_TRAIN_HYBRID - Train GPR model with Matern 3/2 kernel
%
% Inputs:
%   X - Training inputs (n x D)
%   y - Training targets (n x 1)
%
% Optional params:
%   'length_scale' - Initial length scale (default: auto)
%   'signal_var' - Signal variance (default: 1.0)
%   'noise_var' - Noise variance (default: 1e-4)
%   'optimize' - Optimize hyperparameters (default: true)
%
% Output:
%   model - GPR model struct with:
%     .X_train, .y_train
%     .length_scale, .signal_var, .noise_var
%     .L (Cholesky factor), .alpha (solution vector)
%     .log_lik (log marginal likelihood)

    X = double(X);
    y = double(y(:));
    n = size(X, 1);
    dim = size(X, 2);

    % Default parameters
    ls = 1.0;
    sv = 1.0;
    nv = 1e-4;
    optimize = true;

    for i = 1:2:length(varargin)-1
        switch varargin{i}
            case 'length_scale', ls = varargin{i+1};
            case 'signal_var', sv = varargin{i+1};
            case 'noise_var', nv = varargin{i+1};
            case 'optimize', optimize = varargin{i+1};
        end
    end

    model = struct();
    model.X_train = X;
    model.y_train = y;

    % Auto length-scale based on data distribution
    if n > 1
        Dmat = pdist2(X, X, 'euclidean');
        md = median(Dmat(:));
        if md > 1e-6
            ls = max(md * sqrt(dim) / 4, 0.5);  % Scale by sqrt(dim)
        end
    end

    model.length_scale = ls;
    model.signal_var = sv;
    model.noise_var = nv;

    % Optimize hyperparameters if requested
    if optimize && n >= 5
        model = optimize_gpr_hyperparameters(model);
    end

    ls = max(model.length_scale, 1e-6);
    sv = model.signal_var;
    nv = max(model.noise_var, 1e-8);

    % Compute kernel matrix with Matern 3/2
    Dmat = pdist2(X, X, 'euclidean');
    % Matern 3/2 kernel: K(d) = sigma^2 * (1 + sqrt(3)*d/l) * exp(-sqrt(3)*d/l)
    K = sv * (1 + sqrt(3)*Dmat/ls) .* exp(-sqrt(3)*Dmat/ls);
    K = K + nv * eye(n);  % Add noise for numerical stability

    try
        L = chol(K, 'lower');
        alpha = L' \ (L \ y);
        model.L = L;
        model.alpha = alpha;
    catch
        % Fallback with stronger regularization
        K = K + 1e-6 * eye(n);
        L = chol(K, 'lower');
        model.L = L;
        model.alpha = L' \ (L \ y);
    end

    % Compute log marginal likelihood
    model.log_lik = compute_gpr_log_lik(K, model.L, model.alpha, y, nv, n);
    model.is_fitted = true;
end


function [mu, sigma2] = gpr_predict_hybrid(model, X)
% GPR_PREDICT_HYBRID - Predict mean and variance using trained GPR model
%
% Inputs:
%   model - Trained GPR model from gpr_train_hybrid
%   X     - Query points (n_test x D)
%
% Outputs:
%   mu    - Predicted mean (n_test x 1)
%   sigma2 - Predicted variance (n_test x 1)

    X = double(X);

    % Handle single point input
    if size(X, 1) == 1 && size(X, 2) ~= size(model.X_train, 2)
        X = X';
    end

    % Ensure X is n_test x dim
    if size(X, 2) ~= size(model.X_train, 2)
        X = X';
    end

    n_test = size(X, 1);

    % Compute kernel between test and training points
    ls = max(model.length_scale, 1e-6);
    sv = model.signal_var;

    D = pdist2(X, model.X_train, 'euclidean');
    % Matern 3/2 kernel
    K_star = sv * (1 + sqrt(3)*D/ls) .* exp(-sqrt(3)*D/ls);

    % Predictive mean
    mu = K_star * model.alpha;

    % Predictive variance (analytical formula)
    if isfield(model, 'L')
        v = model.L \ K_star';
        sigma2 = sv - sum(v.^2, 1)' + model.noise_var;
        sigma2 = max(sigma2, 1e-8);
    else
        sigma2 = sv * ones(n_test, 1);
    end
end


function model = optimize_gpr_hyperparameters(model)
% OPTIMIZE_GPR_HYPERPARAMETERS - Optimize GPR hyperparameters via grid search

    X = model.X_train;
    y = model.y_train;
    n = size(X, 1);
    dim = size(X, 2);

    % Initialize with auto values
    ls_init = model.length_scale;
    sv_init = model.signal_var;
    nv_init = max(model.noise_var, 1e-4);

    best_ls = ls_init;
    best_sv = sv_init;
    best_nv = nv_init;
    best_ll = -inf;

    % Grid search over hyperparameters
    ls_candidates = ls_init * [0.5, 0.75, 1.0, 1.25, 1.5];
    sv_candidates = sv_init * [0.5, 1.0, 2.0];
    nv_candidates = [1e-5, 1e-4, 1e-3];

    for ls = ls_candidates
        for sv = sv_candidates
            for nv = nv_candidates
                ls = max(ls, 1e-6);
                D = pdist2(X, X, 'euclidean');
                K = sv * (1 + sqrt(3)*D/ls) .* exp(-sqrt(3)*D/ls);
                K = K + nv * eye(n);

                try
                    L = chol(K, 'lower');
                    alpha = L' \ (L \ y);
                    ll = compute_gpr_log_lik(K, L, alpha, y, nv, n);

                    if ll > best_ll
                        best_ll = ll;
                        best_ls = ls;
                        best_sv = sv;
                        best_nv = nv;
                    end
                catch
                    % Skip invalid configurations
                end
            end
        end
    end

    model.length_scale = best_ls;
    model.signal_var = best_sv;
    model.noise_var = best_nv;
end


function ll = compute_gpr_log_lik(K, L, alpha, y, nv, n)
% COMPUTE_GPR_LOG_LIK - Compute log marginal likelihood for GPR

    try
        log_det = 2 * sum(log(diag(L)));
        ll = -0.5 * y' * alpha - 0.5 * log_det - 0.5 * n * log(2*pi);
        ll = max(ll, -1e10);  % Prevent numerical issues
    catch
        ll = -inf;
    end
end


function sigma2 = compute_distance_uncertainty(x_norm, norm_ctx)
% COMPUTE_DISTANCE_UNCERTAINTY - Fallback distance-based uncertainty
% Used when GPR is not available or fails

    if isempty(norm_ctx) || ~isfield(norm_ctx, 'x_ref')
        sigma2 = ones(size(x_norm, 1), 1);
        return;
    end

    x_ref = norm_ctx.x_ref;
    if size(x_ref, 1) < 2
        sigma2 = ones(size(x_norm, 1), 1);
        return;
    end

    % Compute minimum distance to training samples
    D = pdist2(x_norm, x_ref, 'euclidean');
    min_dist = min(D, [], 2);

    % Convert distance to uncertainty (farther = more uncertain)
    sigma2 = min_dist.^2;
    sigma2 = max(sigma2, 1e-6);
end


function [y, sigma2] = hybrid_predict_with_lcb(model, x, kappa)
% HYBRID_PREDICT_WITH_LCB - Predict with LCB acquisition function
%
% LCB(x) = mu(x) - kappa * sigma(x)
%
% Inputs:
%   model - Hybrid surrogate model
%   x     - Query points (nq x D)
%   kappa - Exploration parameter (default: 2.0)
%
% Outputs:
%   lcb   - LCB acquisition values (nq x 1)
%   y     - Predicted mean (nq x 1)
%   sigma - Predicted standard deviation (nq x 1)

    if nargin < 3
        kappa = 2.0;
    end

    [y, sigma2] = hybrid_predict(model, x);
    sigma = sqrt(sigma2);
    lcb = y - kappa * sigma;

    % Return as first output for compatibility
    if nargout <= 1
        y = lcb;
    end
end


function [y, sigma2] = surrogate_predict_with_gpr_uncertainty(model, x)
% SURROGATE_PREDICT_WITH_GPR_UNCERTAINTY - Wrapper for GPR-based uncertainty
% Replaces the earlier heuristic distance-based uncertainty estimator
%
% This function should be used instead of surrogate_predict_mean_uncertainty
% when GPR is available for proper uncertainty quantification.

    if isfield(model, 'hybrid_model') && ~isempty(model.hybrid_model)
        % Use hybrid model
        [y, sigma2] = hybrid_predict(model.hybrid_model, x);
    elseif isfield(model, 'gpr_model') && ~isempty(model.gpr_model)
        % Use standalone GPR
        x_norm = normalize_query_points(x, model.norm_ctx, model.cfg);
        [y, sigma2] = gpr_predict_hybrid(model.gpr_model, x_norm);
    else
        % Fallback to original heuristic uncertainty
        [y, sigma2] = surrogate_predict_mean_uncertainty(model, x);
    end
end

%% ============================================================================
% TRIPLE-KERNEL RBF ENSEMBLE SURROGATE MODEL
% Multi-kernel RBF ensemble based on DS-SAEA (Ma et al., Applied Soft Computing 2025)
% - Three RBF kernels: Multi-Quadric (MQ), Thin-Plate Spline (TPS), Gaussian
% - Adaptive kernel weighting based on inverse training RMSE
% - Dimension-aware epsilon scaling for high-dimensional problems
% ============================================================================

