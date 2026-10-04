%%%*********************************************************************************************%%%
%% Benchmark functions for SAPSO-ERD
%% H. Yu, Y. Tan, J. Zeng, C. Sun, Y. Jin, Surrogate-assisted hierarchical 
%% particle swarm optimization, Information Sciences, 454-455 (2018) 59-72.
%%%*********************************************************************************************%%%
%% This paper and this code should be referenced whenever they are used to 
%% generate results for the user's own research. 
%%%*********************************************************************************************%%%
%% This matlab code was written by Haibo Yu
%% Please refer with all questions, comments, bug reports, etc. to tyustyuhaibo@126.com
% 
%% Test functions for category: 'FITNESS'

function [y] = FITNESS(xx, func_id)
% FITNESS benchmark suite selector
% func_id:
%   1 - Ackley
%   2 - Griewank (default, backward compatible)
%   3 - Rosenbrock
%   4 - Ellipsoid
%   5 - Rastrigin
%   6 - CEC2005 F10: Shifted Rotated Rastrigin
%   7 - CEC2005 F19: Rotated Hybrid Composition Function

if nargin < 2 || isempty(func_id)
    func_id = 2;
end

switch func_id
    case 1
        y = ackley_func(xx);
    case 2
        y = griewank_func(xx);
    case 3
        y = rosenbrock_func(xx);
    case 4
        y = ellipsoid_func(xx);
    case 5
        y = rastrigin_func(xx);
    case 6
        y = cec2005_comparison_eval(xx, 10);
    case 7
        y = cec2005_comparison_eval(xx, 19);
    otherwise
        error('FITNESS:InvalidFunctionId', ...
            'Unsupported func_id=%d. Valid ids are 1,2,3,4,5,6,7.', func_id);
end
end

function y = ackley_func(xx)
d = length(xx);
a = 20;
b = 0.2;
c = 2*pi;
sum1 = 0;
sum2 = 0;
for ii = 1:d
    xi = xx(ii);
    sum1 = sum1 + xi^2;
    sum2 = sum2 + cos(c * xi);
end
term1 = -a * exp(-b * sqrt(sum1 / d));
term2 = -exp(sum2 / d);
y = term1 + term2 + a + exp(1);
end

function y = griewank_func(xx)
d = length(xx);
sum1 = 0;
prod1 = 1;
for ii = 1:d
    xi = xx(ii);
    sum1 = sum1 + xi^2 / 4000;
    prod1 = prod1 * cos(xi / sqrt(ii));
end
y = sum1 - prod1 + 1;
end

function y = rosenbrock_func(xx)
d = length(xx);
sum1 = 0;
for ii = 1:(d-1)
    xi = xx(ii);
    xnext = xx(ii + 1);
    sum1 = sum1 + 100 * (xnext - xi^2)^2 + (xi - 1)^2;
end
y = sum1;
end

function y = ellipsoid_func(xx)
d = length(xx);
sum1 = 0;
for ii = 1:d
    xi = xx(ii);
    sum1 = sum1 + ii * xi^2;
end
y = sum1;
end

function y = rastrigin_func(xx)
d = length(xx);
sum1 = 0;
for ii = 1:d
    xi = xx(ii);
    sum1 = sum1 + (xi^2 - 10 * cos(2 * pi * xi));
end
y = 10 * d + sum1;
end
