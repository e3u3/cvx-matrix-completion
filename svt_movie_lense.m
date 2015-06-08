%% Matrix completion with SVT
%% Load data
load('../data/movie_lense_100k.mat');
movie_Y = Y;
movie_R = R;
disp(size(movie_Y))
clear Y
clear R
crop = 100;
if crop > 0
    movie_Y = movie_Y(1:crop, 1:crop);
    movie_R = movie_R(1:crop, 1:crop);
end

%% Setup a problem
rng(234923);
N = size(movie_Y, 1);
M = size(movie_Y, 2);
shuffler = randperm(N);
movie_Y = movie_Y(shuffler, :);
movie_R = movie_R(shuffler, :);

%% Setup mask
train_portion = 0.95;
visible_ind = find(movie_R == 1)';
visible_num = length(visible_ind);
train_visible_num = round(visible_num * train_portion);
rPerm = randperm(visible_num);
omega = visible_ind(sort(rPerm(1:train_visible_num)));

%% Matrix completion via TFOCS
myPath = '/home/linuxthink/Dropbox/courses/ece273/project/packages/tfocs/TFOCS-1.3.1/';
if exist(myPath, 'dir')
    addpath(myPath)
end

observations = movie_Y(omega);    % the observed entries
mu           = .001;        % smoothing parameter

% The solver runs in seconds
tic
[ Xk, out, opts ] = solver_sNuclearBP( {N,M,omega}, observations, mu );
toc

test_ind = setdiff(visible_ind, omega);
diff = Xk - movie_Y;
diff = diff(test_ind);
rmse = sqrt(sum(sum(diff .^ 2)) / length(test_ind));
disp(rmse);