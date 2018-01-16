import numpy as np

def next_state(s, a, t):
    # calculate the next state
    s_next = np.dot(A, s) + np.dot(B, a) + np.diag(np.random.normal(b, np.sqrt(cov))).reshape((2, 1))
    return s_next.squeeze()

def control_signal(s, s_des, t, K_t, k_t):
    # calculate the action
    a = np.dot(K_t, s_des - s) + k_t
    return a

def reward(s, a, t):
    rew = 0
    R = get_R(t)
    r = get_r(t)

    if t < T:
        rew = -np.dot(np.dot((s - r).T, R), (s - r)) - np.dot(np.dot(a.T, H), a)
    elif t == T:
        rew = -np.dot(np.dot((s - r).T, R), (s - r))
    else:
        rew = 'error'
    return rew

def run_system(s_des, K_t=0, k_t=k):
    if K_t == 0:
        K_t = K

    s = np.zeros((2, T + 1))
    a = np.zeros(T + 1)
    r = np.zeros(T + 1)
    s[:, 0] = np.diag(np.random.normal(0, np.eye(2)))
    a[0] = control_signal(s[:, 0].reshape(2, 1), s_des[:, 0].reshape(2, 1), 0, K_t, k_t)
    for t in range(T):
        r[t] = reward(s[:, t].reshape(2, 1), a[t], t)
        s[:, t + 1] = next_state(s[:, t].reshape(2, 1), a[t], t)
        a[t + 1] = control_signal(s[:, t + 1].reshape(2, 1), s_des[:, t + 1].reshape(2, 1), t + 1, K_t, k_t)

    r[T] = reward(s[:, T].reshape(2, 1), a[T], T)
    return s, a, r
