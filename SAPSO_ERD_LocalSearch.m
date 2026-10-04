function local_pop_updated = SAPSO_ERD_LocalSearch(local_pop, fitness, Dim, gap, K, LB, UB, ls_config)
% Local search for SAPSO-ERD with interval grouping and optional enhanced update policy.
% local_pop: N x D
% fitness:   surrogate fitness, kept for API compatibility

if nargin < 7
    error('SAPSO_ERD_LocalSearch requires local_pop, fitness, Dim, gap, K, LB, UB.');
end
if nargin < 8 || isempty(ls_config)
    ls_config = default_ls_config();
end
if ~isfield(ls_config, 'mode')
    ls_config.mode = 'enhanced';
end

if isempty(local_pop)
    local_pop_updated = local_pop;
    return;
end

[num_particles, num_dims] = size(local_pop);
if num_dims ~= Dim
    error('Dim mismatch: size(local_pop,2) must equal Dim.');
end
if numel(LB) ~= Dim || numel(UB) ~= Dim
    error('LB/UB size mismatch: both must be 1 x Dim.');
end

fitness = fitness(:); %#ok<NASGU>
LB = reshape(LB, 1, []);
UB = reshape(UB, 1, []);

[sorted_intervals, sorted_index, dim_bounds] = build_intervals(local_pop);
[group_dims, group_intervals] = cluster_overlap_groups(sorted_intervals, sorted_index);
[group_dims, group_intervals] = merge_small_groups(group_dims, group_intervals, K);
group_intervals = expand_intervals(group_intervals, gap, LB, UB);

local_pop_updated = local_pop;
if isempty(group_dims)
    return;
end

switch lower(ls_config.mode)
    case 'legacy'
        local_pop_updated = apply_legacy_update(local_pop_updated, group_dims, group_intervals, Dim, K, LB, UB);
    otherwise
        local_pop_updated = apply_enhanced_update(local_pop_updated, group_dims, group_intervals, dim_bounds, K, LB, UB, ls_config);
end
end

function cfg = default_ls_config()
cfg.mode = 'enhanced';
cfg.center_pull_fine = 0.45;
cfg.center_pull_coarse = 0.25;
cfg.noise_ratio_fine = 0.08;
cfg.noise_ratio_coarse = 0.15;
cfg.use_levy = false;
cfg.levy_prob = 0.2;
cfg.levy_beta = 1.5;
cfg.levy_scale_fine = 0.05;
cfg.levy_scale_coarse = 0.10;
end

function pop_updated = apply_legacy_update(pop_updated, group_dims, group_intervals, Dim, K, LB, UB)
num_particles = size(pop_updated, 1);
for p = 1:num_particles
    x = pop_updated(p, :);
    v = randn(1, Dim) * 0.5;
    for g = 1:numel(group_dims)
        dims = group_dims{g};
        if numel(dims) >= K
            step_scale = 0.3;
        else
            step_scale = 1.0;
        end
        x(dims) = x(dims) + step_scale * v(dims);
        x(dims) = max(x(dims), group_intervals(g, 1));
        x(dims) = min(x(dims), group_intervals(g, 2));
    end
    x = max(x, LB);
    x = min(x, UB);
    pop_updated(p, :) = x;
end
end

function pop_updated = apply_enhanced_update(pop_updated, group_dims, group_intervals, dim_bounds, K, LB, UB, ls_config)
num_particles = size(pop_updated, 1);
if ~isfield(ls_config, 'use_levy'); ls_config.use_levy = false; end
if ~isfield(ls_config, 'levy_prob'); ls_config.levy_prob = 0.2; end
if ~isfield(ls_config, 'levy_beta'); ls_config.levy_beta = 1.5; end
if ~isfield(ls_config, 'levy_scale_fine'); ls_config.levy_scale_fine = 0.05; end
if ~isfield(ls_config, 'levy_scale_coarse'); ls_config.levy_scale_coarse = 0.10; end

for p = 1:num_particles
    x = pop_updated(p, :);
    for g = 1:numel(group_dims)
        dims = group_dims{g};
        num_dims_in_group = numel(dims);
        if num_dims_in_group >= K
            center_pull = ls_config.center_pull_fine;
            noise_ratio = ls_config.noise_ratio_fine;
            levy_scale = ls_config.levy_scale_fine;
        else
            center_pull = ls_config.center_pull_coarse;
            noise_ratio = ls_config.noise_ratio_coarse;
            levy_scale = ls_config.levy_scale_coarse;
        end

        group_lb = group_intervals(g, 1);
        group_ub = group_intervals(g, 2);

        for j = 1:num_dims_in_group
            d = dims(j);
            lb_d = max(group_lb, dim_bounds(d, 1));
            ub_d = min(group_ub, dim_bounds(d, 2));
            if ub_d <= lb_d
                lb_d = group_lb;
                ub_d = group_ub;
            end

            width_d = max(ub_d - lb_d, 1e-12);
            center_d = 0.5 * (lb_d + ub_d);
            noise = noise_ratio * width_d * randn;
            step = center_pull * (center_d - x(d)) + noise;
            if ls_config.use_levy && rand < ls_config.levy_prob
                step = step + levy_scale * width_d * levy_step(ls_config.levy_beta);
            end
            x(d) = x(d) + step;
            x(d) = max(x(d), lb_d);
            x(d) = min(x(d), ub_d);
        end
    end
    x = max(x, LB);
    x = min(x, UB);
    pop_updated(p, :) = x;
end

function s = levy_step(beta)
% Mantegna algorithm for symmetric Levy alpha-stable steps.
sigma_u = (gamma(1+beta) * sin(pi*beta/2) / ...
    (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta);
u = sigma_u * randn;
v = randn;
s = u / (abs(v)^(1/beta) + 1e-12);
end
end

function [sorted_intervals, sorted_index, dim_bounds] = build_intervals(local_pop)
dim_lb = min(local_pop, [], 1);
dim_ub = max(local_pop, [], 1);
dim_bounds = [dim_lb(:), dim_ub(:)];
lu = [dim_lb; dim_ub];
[sorted_intervals, sorted_index] = sortrows(lu', 1);
end

function [group_dims, group_intervals] = cluster_overlap_groups(sorted_intervals, sorted_index)
num_intervals = size(sorted_intervals, 1);
group_dims = {};
group_intervals = zeros(num_intervals, 2);

group_count = 1;
current_interval = sorted_intervals(1, :);
current_dims = sorted_index(1);

for i = 2:num_intervals
    next_interval = sorted_intervals(i, :);
    overlap_interval = [max(current_interval(1), next_interval(1)), ...
                        min(current_interval(2), next_interval(2))];
    if overlap_interval(1) <= overlap_interval(2)
        current_interval = overlap_interval;
        current_dims(end + 1) = sorted_index(i); %#ok<AGROW>
    else
        group_dims{group_count} = current_dims; %#ok<AGROW>
        group_intervals(group_count, :) = current_interval;
        group_count = group_count + 1;
        current_interval = next_interval;
        current_dims = sorted_index(i);
    end
end

group_dims{group_count} = current_dims;
group_intervals(group_count, :) = current_interval;
group_intervals = group_intervals(1:group_count, :);
end

function [group_dims, group_intervals] = merge_small_groups(group_dims, group_intervals, K)
num_groups = numel(group_dims);
if num_groups <= 1
    return;
end

is_active = true(num_groups, 1);
centers = mean(group_intervals, 2);
widths = group_intervals(:, 2) - group_intervals(:, 1);

for i = 1:num_groups
    if ~is_active(i)
        continue;
    end
    if numel(group_dims{i}) >= K
        continue;
    end

    candidates = find(is_active);
    candidates(candidates == i) = [];
    if isempty(candidates)
        continue;
    end

    dists = abs(centers(candidates) - centers(i));
    min_dist = min(dists);
    tie_mask = abs(dists - min_dist) < 1e-12;
    tie_candidates = candidates(tie_mask);

    if numel(tie_candidates) > 1
        [~, idx] = max(widths(tie_candidates));
        target = tie_candidates(idx);
    else
        target = tie_candidates(1);
    end

    group_dims{target} = unique([group_dims{target}, group_dims{i}], 'stable');
    is_active(i) = false;
end

group_dims = group_dims(is_active);
group_intervals = group_intervals(is_active, :);
end

function expanded_intervals = expand_intervals(group_intervals, gap, LB, UB)
expanded_intervals = group_intervals;
global_lb = min(LB);
global_ub = max(UB);

for g = 1:size(group_intervals, 1)
    width = expanded_intervals(g, 2) - expanded_intervals(g, 1);
    if width < gap
        mid = 0.5 * (expanded_intervals(g, 1) + expanded_intervals(g, 2));
        expanded_intervals(g, 1) = mid - 0.5 * gap;
        expanded_intervals(g, 2) = mid + 0.5 * gap;
    end
    expanded_intervals(g, 1) = max(expanded_intervals(g, 1), global_lb);
    expanded_intervals(g, 2) = min(expanded_intervals(g, 2), global_ub);
end
end
