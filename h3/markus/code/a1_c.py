class calcOptK:
    def __init__(self):
        self.K_t = nan_matrix((2, T + 1))
        self.k_t = nan_matrix(T + 1)
        self.M_t = nan_matrix((2, 2, T + 1))
        self.V_t = nan_matrix((2, 2, T + 1))
        self.v_t = nan_matrix((2, T + 1))

    def getOptReg(self):
        for i in range(T):
            self.k_t[i] = self.get_k(i)
            self.K_t[:, i] = self.get_K(i)
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
        if np.isnan(self.V_t[:, :, t]).all():
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
            return self.V_t[:, :, t]

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
            return self.v_t[:, t].reshape(2, 1)

    def get_K(self, t):
        V_next = self.get_V(t + 1)
        temp1 = inv(H + np.dot(np.dot(B.T, V_next), B))
        temp2 = np.dot(np.dot(B.T, V_next), A)
        K = np.dot(temp1, temp2)  
        return K

    def get_k(self, t):
        V_next = self.get_V(t + 1)
        v_next = self.get_v(t + 1)
        temp1 = inv(H + np.dot(np.dot(B.T, V_next), B))
        temp2 = np.dot(B.T, (np.dot(V_next, b) - v_next))
        k = -np.dot(temp1, temp2)
        return k.squeeze()

    def get_R(self, t):
        if t == 14 or t == 40:
            R = R1
        else:
            R = R2
        return R

    def get_r(self, t):
        if t <= 14:
            r = r1
        else:
            r = r2
        return r
