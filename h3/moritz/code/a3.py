import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

T = 50
A = np.array([[1, 0.1], [0, 1]])
B = np.array([[0], [0.1]])
b = np.array([[5], [0]]) #np.zeros((2,1))#
cov = np.array([[0.01, 0], [0, 0.01]])
K = np.array([5, 0.3]) #np.array([24.5, 9.9]) #
k = 0.3
H = 1
R1 = np.array([[100000, 0], [0, 0.1]])  # t=14 or 40
R2 = np.array([[0.01, 0], [0, 0.1]])  # otherwise
r1 = np.array([[10], [0]])  # t = 0...14
r2 = np.array([[20], [0]])  # t = 15...T


def next_state(s, a, t):
    s_next = np.dot(A, s) + np.dot(B, a) + np.diag(np.random.normal(b, np.sqrt(cov))).reshape((2,1))
    return s_next.squeeze()


def control_signal(s, s_des, t, K_t, k_t):
    a = np.dot(K_t, s_des-s) + k_t
    return a

def get_R(t):
    if t == 14 or t == 40:
        R = R1
    else:
        R = R2
    return R

def get_r(t):
    if t <= 14:
        r = r1
    else:
        r = r2
    return r


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

    s = np.zeros((2, T+1))
    a = np.zeros(T+1)
    r = np.zeros(T+1)
    s[:, 0] = np.diag(np.random.normal(0, np.eye(2)))
    a[0] = control_signal(s[:, 0].reshape(2,1), s_des[:, 0].reshape(2,1), 0, K_t, k_t)
    for t in xrange(T):
        r[t] = reward(s[:,t].reshape(2,1), a[t], t)
        s[:,t+1] = next_state(s[:, t].reshape(2,1), a[t], t)
        a[t+1] = control_signal(s[:, t+1].reshape(2,1), s_des[:, t+1].reshape(2,1), t+1, K_t, k_t)

    r[T] = reward(s[:, T].reshape(2,1), a[T], T)
    return s,a, r


def plot_system(s, a, r, s_des, fig_num):
    nr_subplots = 4
    s_mean = np.mean(s, 2)
    a_mean = np.mean(a, 1)
    r_mean = np.mean(r, 1)

    s_std = np.sqrt(np.var(s, 2))
    a_std = np.sqrt(np.var(a, 1))
    r_std = np.sqrt(np.var(r, 1))

    r_sum = np.sum(r, axis=0)
    r_sum_mean = np.mean(r_sum)
    r_sum_std = np.sqrt(np.var(r_sum))
    print 'r_sum_mean: {}'.format(r_sum_mean)
    print 'r_sum_std: {}'.format(r_sum_std)

    t = range(T+1)

    plt.figure(fig_num)
    grey = [0.7, 0.7, 0.7]
    """
    for i in xrange(nr_runs):
        plt.subplot(3, 1, 1)
        plt.plot(a[:,i], color=grey, alpha=0.5)

        plt.subplot(3, 1, 2)
        plt.plot(s[0, :, i], color=grey, alpha=0.5)

        plt.subplot(3, 1, 3)
        plt.plot(s[1, :, i], color=grey, alpha=0.5)
        plt.xlabel('t')
    """

    plt.subplot(nr_subplots, 1, 1)
    plt.plot(a_mean)
    plt.fill_between(t, a_mean + 2 * a_std, a_mean - 2 * a_std,
                     alpha=0.5)

    plt.subplot(nr_subplots, 1, 2)
    plt.plot(t, s_mean[0, :])
    plt.fill_between(t, s_mean[0, :] + 2 * s_std[0, :], s_mean[0, :] - 2 * s_std[0, :],
                     alpha=0.5)
    plt.plot(t, s_des[0,:], color='k')

    plt.subplot(nr_subplots, 1, 3)
    plt.plot(t, s_mean[1, :])
    plt.fill_between(t, s_mean[1, :] + 2 * s_std[1, :], s_mean[1, :] - 2 * s_std[1, :],
                     alpha=0.5)
    plt.plot(t, s_des[1, :], color='k')

    plt.subplot(nr_subplots, 1, 4)
    plt.plot(t, r_mean)
    plt.fill_between(t, r_mean + 2 * r_std, r_mean - 2 * r_std,
                     alpha=0.5)
    plt.xlabel('t')

    return s_mean, s_std


def nan_matrix(shape):
    """
    :param shape: shape tupel
    :return: Matrix with NAN values
    """
    m = np.empty(shape)
    m.fill(np.nan)
    return m


class calcOptK:
    def __init__(self):
        self.K_t = nan_matrix((2, T+1))
        self.k_t = nan_matrix(T+1)
        self.M_t = nan_matrix((2,2,T+1))
        self.V_t = nan_matrix((2,2,T+1))
        self.v_t = nan_matrix((2,T+1))

    def getOptReg(self):
        for i in xrange(T):
            self.k_t[i] = self.get_k(i)
            self.K_t[:,i] = self.get_K(i)
        return self.K_t, self.k_t

    def get_M(self, t):
        if np.isnan(self.M_t[:, :, t]).all():
            V_next = self.get_V(t + 1)
            temp1 = inv(H + np.dot(np.dot(B.T, V_next), B))
            temp2 = np.dot(np.dot(B.T, V_next), A)
            M = np.dot(np.dot(B, temp1), temp2)
            self.M_t[:, :, t] = M
            return M
        else:
            return self.M_t[:, :, t]

    def get_V(self, t):
        if np.isnan(self.V_t[:,:,t]).all():
            R = self.get_R(t)
            if t < T:
                V_next = self.get_V(t + 1)
                M = self.get_M(t)
                V = R + np.dot(np.dot((A - M).T, V_next), A)
            elif t == T:
                V = R
            else:
                V = 'error'
            self.V_t[:, :, t] = V
            return V
        else:
            return self.V_t[:,:,t]

    def get_v(self, t):
        if np.isnan(self.v_t[:, t]).all():
            R = self.get_R(t)
            r = self.get_r(t)

            if t < T:
                M = self.get_M(t)
                v_next = self.get_v(t + 1)
                V_next = self.get_V(t + 1)
                v = np.dot(R, r) + np.dot((A - M).T, (v_next - np.dot(V_next, b)))
            elif t == T:
                v = np.dot(R, r)
            else:
                v = 'error'
            self.v_t[:, t] = v.squeeze()
            return v
        else:
            return self.v_t[:,t].reshape(2,1)

    def get_K(self, t):
        V_next = self.get_V(t + 1)
        temp1 = inv(H + np.dot(np.dot(B.T, V_next), B))
        temp2 = np.dot(np.dot(B.T, V_next), A)
        K = np.dot(temp1, temp2) # TODO minus!
        return K

    def get_k(self, t):
        V_next = self.get_V(t + 1)
        v_next = self.get_v(t + 1)
        temp1 = inv(H + np.dot(np.dot(B.T, V_next), B))
        temp2 = np.dot(B.T, (np.dot(V_next, b) - v_next))
        k = -np.dot(temp1, temp2) # TODO minus!
        return k.squeeze()

    def get_R(self, t):
        if t == 14 or t == 40:
            R = R1
        else:
            R = R2
        return R

    def get_r(self, t):
        if t <= 14:
            r = r1 #TODO minus
        else:
            r = r2 #TODO minus
        return r


def run_system_Kt(s_des, K_t, k_t):
    s = np.zeros((2, T+1))
    a = np.zeros(T+1)
    r = np.zeros(T+1)
    s[:, 0] = np.diag(np.random.normal(0, np.eye(2)))
    a[0] = control_signal(s[:, 0].reshape(2,1), s_des[:, 0].reshape(2,1), 0, K_t[:,0], k_t[0])
    for t in xrange(T):
        r[t] = reward(s[:,t].reshape(2,1), a[t], t)
        s[:,t+1] = next_state(s[:, t].reshape(2,1), a[t], t)
        a[t+1] = control_signal(s[:, t+1].reshape(2,1), s_des[:, t+1].reshape(2,1), t+1, K_t[:,t+1], k_t[t+1])

    r[T] = reward(s[:, T].reshape(2,1), a[T], T)
    return s,a, r


def main():
    # 3.1 a)
    # one run as test
    s1, a1, r1 = run_system(np.zeros((2,T+1)))

    nr_subplots = 4
    plt.figure(1)
    plt.subplot(nr_subplots,1,1)
    plt.plot(a1, label='a')
    plt.legend()

    plt.subplot(nr_subplots,1,2)
    plt.plot(s1[0,:], label='s1')
    plt.legend()

    plt.subplot(nr_subplots, 1, 3)
    plt.plot(s1[1, :], label='s2')
    plt.xlabel('t')
    plt.legend()

    plt.subplot(nr_subplots, 1, 4)
    plt.plot(r1, label='r')
    plt.xlabel('t')
    plt.legend()

    plt.subplot(nr_subplots,1,1)
    plt.title('3.1 a) one run')


    # run system 20 times
    nr_runs = 20
    s2 = np.zeros((2,T+1,nr_runs))
    a2 = np.zeros((T+1,nr_runs))
    r2 = np.zeros((T+1, nr_runs))
    
    for i in xrange(nr_runs):
        s2[:,:,i], a2[:,i], r2[:,i] = run_system(np.zeros((2,T+1)))

    s1_mean, s1_std = plot_system(s2, a2, r2, np.zeros((2,T+1)), 2)
    plt.subplot(nr_subplots, 1, 1)
    plt.title('3.1 a) 20 times')

    # 3.1 b)
    temp1 = np.array([[np.ones([1,15])], [np.zeros([1,15])]]).squeeze()
    temp2 = np.array([[np.ones([1, T+1-15])], [np.zeros([1, T+1-15])]]).squeeze()
    s_des = np.append(10.*temp1, 20.*temp2, axis=1)

    s3 = np.zeros((2, T+1, nr_runs))
    a3 = np.zeros((T+1, nr_runs))
    r3 = np.zeros((T+1, nr_runs))

    for i in xrange(nr_runs):
        s3[:, :, i], a3[:, i], r3[:, i] = run_system(s_des)

    s2_mean, s2_std = plot_system(s3, a3, r3, s_des, 3)
    plt.subplot(nr_subplots, 1, 1)
    plt.title('3.1 b) 20 times with s_des = r')

    t = range(T+1)
    plt.figure(4)
    plt.subplot(2,1,1)
    plt.plot(t, s1_mean[0, :])
    plt.fill_between(t, s1_mean[0, :] + 2 * s1_std[0, :], s1_mean[0, :] - 2 * s1_std[0, :],
                     alpha=0.5)
    plt.plot(t, s2_mean[0, :])
    plt.fill_between(t, s2_mean[0, :] + 2 * s2_std[0, :], s2_mean[0, :] - 2 * s2_std[0, :],
                     alpha=0.5)

    plt.subplot(2,1,2)
    plt.plot(t, s1_mean[1, :])
    plt.fill_between(t, s1_mean[1, :] + 2 * s1_std[1, :], s1_mean[1, :] - 2 * s1_std[1, :],
                     alpha=0.5)
    plt.plot(t, s2_mean[1, :])
    plt.fill_between(t, s2_mean[1, :] + 2 * s2_std[1, :], s2_mean[1, :] - 2 * s2_std[1, :],
                     alpha=0.5)

    plt.subplot(2,1,1)
    plt.title('3.1 b) 20 times with s_des = r and s_des = 0')

    # 3.1 c)
    opt = calcOptK()
    K_t, k_t = opt.getOptReg()

    plt.figure(5)
    plt.subplot(2,1,1)
    plt.plot(k_t)
    plt.subplot(2, 1, 2)
    plt.plot(K_t.T)
    plt.subplot(2, 1, 1)
    plt.title('3.1 c) K and k')

    K_t[:,T]=K_t[:,T-1]
    k_t[T] = k_t[T - 1]

    s4 = np.zeros((2, T+1, nr_runs))
    a4 = np.zeros((T+1, nr_runs))
    r4 = np.zeros((T+1, nr_runs))

    for i in xrange(nr_runs):
        s4[:, :, i], a4[:, i], r4[:, i] = run_system_Kt(np.zeros((2,T+1)), K_t, k_t)#run_system_Kt(s_des, K_t, k_t)

    s4_mean, s4_std = plot_system(s4, a4, r4, np.zeros((2,T+1)), 6)
    plt.subplot(nr_subplots, 1, 1)
    plt.title('3.1 c) 20 times with s_des = 0')

    plt.show()
    return 0


if __name__ == '__main__':
    main()
