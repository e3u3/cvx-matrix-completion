# from __future__ import print_function
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict
from scipy.optimize import fmin_l_bfgs_b as lbfgs
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
from scipy.optimize import minimize
import time

# load data
data = np.load('../data/data.npz')['arr_0']
# data = data[0:1000, :]
data_train, data_test = train_test_split(data, train_size=0.95, random_state=1)
data_train = data_train.astype('int64')
data_test = data_test.astype('int64')
user_ids = np.unique(data_train[:, 0])
item_ids = np.unique(data_train[:, 1])
user_num = user_ids.shape[0]
item_num = item_ids.shape[0]
alpha = np.mean(data_train[:, 2])

def rmse(gt_label, pd_ratings):
    x = (gt_label.astype('float') - pd_ratings.astype('float')) ** 2.
    mse = np.sum(x) / x.shape[0]
    rmse = mse ** 0.5
    return rmse

def pack(beta_users, beta_items, gamma_users, gamma_items):
    return np.concatenate((beta_users.ravel(),
                           beta_items.ravel(),
                           gamma_users.ravel(),
                           gamma_items.ravel()))

def unpack(theta):
    curr_ind = 0
    beta_users = theta[curr_ind : curr_ind + user_num]
    curr_ind += user_num
    beta_items = theta[curr_ind : curr_ind + item_num]
    curr_ind += item_num
    gamma_users = theta[curr_ind : curr_ind + K * user_num].reshape((K, user_num))
    curr_ind += K * user_num
    gamma_items = theta[curr_ind :].reshape((K, item_num))
    return [beta_users, beta_items, gamma_users, gamma_items]

def init_theta(K):
    beta_users = np.random.normal(0, 0.5, (user_num, ))
    beta_items = np.random.normal(0, 0.5, (item_num, ))
    gamma_users = np.random.normal(0, 0.5, (K, user_num))
    gamma_items = np.random.normal(0, 0.5, (K, item_num))
    return pack(beta_users, beta_items, gamma_users, gamma_items)

# build id_to_sorted_ind
user_id_to_sorted_ind = {}
for i in range(0, user_num):
    user_id_to_sorted_ind[user_ids[i]] = i
item_id_to_sorted_ind = {}
for i in range(0, item_num):
    item_id_to_sorted_ind[item_ids[i]] = i
# write user_ind and item_ind to train_data
data_train = np.hstack((data_train, np.zeros((data_train.shape[0], 2)))).astype('int64')
for i in range(0, data_train.shape[0]):
    datum = data_train[i, :]
    user_ind = user_id_to_sorted_ind[datum[0]]
    item_ind = item_id_to_sorted_ind[datum[1]]
    data_train[i, 3] = user_ind
    data_train[i, 4] = item_ind

# set parameters
K = 10
phi = 0.02
theta = init_theta(K) # all parameters

# sanity check pack and unpack
[a, b, c, d] = unpack(theta)
theta_new = pack(a,b,c,d)
assert(np.array_equal(theta, theta_new))

def objective(theta):
    [beta_users, beta_items, gamma_users, gamma_items] = unpack(theta)
    cost = 0
    for datum in data_train:
        # user_ind = user_id_to_sorted_ind[datum[0]]
        # item_ind = item_id_to_sorted_ind[datum[1]]
        user_ind = datum[3]
        item_ind = datum[4]
        cost += (float(alpha) + beta_users[user_ind] + beta_items[item_ind] + np.dot(gamma_users[:, user_ind], gamma_items[:, item_ind]) - float(datum[2])) ** 2.0
    cost += phi * (np.linalg.norm(theta) ** 2.0)
    return cost

def gradient(theta):
    [beta_users, beta_items, gamma_users, gamma_items] = unpack(theta)
    beta_users_grad = np.zeros((user_num, ))
    beta_items_grad = np.zeros((item_num, ))
    gamma_users_grad = np.zeros((K, user_num))
    gamma_items_grad = np.zeros((K, item_num))
    for datum in data_train:
        # user_ind = user_id_to_sorted_ind[datum[0]]
        # item_ind = item_id_to_sorted_ind[datum[1]]
        user_ind = datum[3]
        item_ind = datum[4]
        prediction = float(alpha) + beta_users[user_ind] + beta_items[item_ind] + np.dot(gamma_users[:, user_ind], gamma_items[:, item_ind])
        common_offset = (prediction - float(datum[2]))
        beta_users_grad[user_ind] += common_offset
        beta_items_grad[item_ind] += common_offset
        gamma_users_grad[:, user_ind] += common_offset * gamma_items[:, item_ind]
        gamma_items_grad[:, item_ind] += common_offset * gamma_users[:, user_ind]
    grad = pack(beta_users_grad, beta_items_grad, gamma_users_grad, gamma_items_grad)
    grad = grad + phi * theta
    grad = grad * 2.
    return grad

def check_gradient(func, grad, x0, *args, **kwargs):
    _epsilon = np.sqrt(np.finfo(float).eps)
    step = kwargs.pop('epsilon', _epsilon)
    if kwargs:
        raise ValueError("Unknown keyword arguments: %r" %
                         (list(kwargs.keys()),))
    numerical = approx_fprime(x0, func, step, *args)
    functional = grad(x0, *args)
    # return np.sqrt(sum((functional - numerical)**2)) # scipy's method
    return np.linalg.norm(numerical-functional) / np.linalg.norm(numerical+functional) # andrew's method

def evaluate(user_id, item_id, theta):
    user_id = int(user_id)
    item_id = int(item_id)
    [beta_users, beta_items, gamma_users, gamma_items] = unpack(theta)
    if user_id in user_id_to_sorted_ind:
        beta_user = beta_users[user_id_to_sorted_ind[user_id]]
        gamma_user = gamma_users[:, user_id_to_sorted_ind[user_id]]
    else:
        beta_user = 0.0
        gamma_user = np.zeros((K, ))
    if item_id in item_id_to_sorted_ind:
        beta_item = beta_items[item_id_to_sorted_ind[item_id]]
        gamma_item = gamma_items[:, item_id_to_sorted_ind[item_id]]
    else:
        beta_item = 0.0
        gamma_item = np.zeros((K, ))
    return alpha + beta_user + beta_item + np.dot(gamma_user, gamma_item)

def predict(target_data, theta):
    ratings = np.zeros((target_data.shape[0], ))
    for i in range(0, target_data.shape[0]):
        ratings[i] = evaluate(target_data[i, 0], target_data[i, 1], theta)
    return ratings

def test_and_get_mse(target_data, theta):
    pd_ratings = predict(target_data[:, 0:0+2], theta).astype('float')
    gt_ratings = target_data[:, 2].astype('float')
    return rmse(pd_ratings, gt_ratings)

start_time = time.time()
def print_progress(curr_theta):
    print "train rmse:", test_and_get_mse(data_train, curr_theta)
    print "test  rmse:", test_and_get_mse(data_test, curr_theta)
    print objective(curr_theta), time.time() - start_time

# [x, f, d] = lbfgs(objective, theta, fprime=gradient, callback=print_progress)
res = minimize(objective, theta, method='L-BFGS-B', jac=gradient, options={'disp': True, 'maxiter': 200},
               callback=print_progress)
