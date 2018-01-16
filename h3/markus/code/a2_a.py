def prob_model(row, col, action, use_prob_model=False):
    if not use_prob_model:
        if action == 0:
            # down
            row_next = row + 1
            col_next = col
            if row_next > 8:
                row_next = row
        elif action == 1:
            # right
            row_next = row
            col_next = col + 1
            if col_next > 9:
                col_next = col
        elif action == 2:
            # up
            row_next = row - 1
            col_next = col
            if row_next < 0:
                row_next = row
        elif action == 3:
            # left
            row_next = row
            col_next = col - 1
            if col_next < 0:
                col_next = col
        elif action == 4:
            # prep -> stay.
            row_next = row
            col_next = col
        return row_next, col_next


def ValIter(R, discount=None, maxSteps=None, infHor=False, probModel=False):
    #R = grid_world matrix! state metric for reward
    if infHor==False:
        T = maxSteps
        # down, right, up, left, prep
        actions = np.array([0, 1, 2, 3, 4])
        Q = np.zeros((R.shape[0], R.shape[1], actions.shape[0], T))
        V = np.zeros((R.shape[0], R.shape[1], T))
        # r = np.zeros((R.shape[0], R.shape[1], actions.shape[0], T))
        # r[:,:,T-1] = R
        V[:, :, T - 1] = R
        # iterate over all ts
        # iterate backwards in time!
        for t in range(T - 2, -1, -1):
            # iterate over all state rows
            for row in range(R.shape[0]):
                # iterate over all columns
                for col in range(R.shape[1]):
                    # iterate over all actions
                    for action in range(actions.shape[0]):
                        if not probModel:
                            row_next, col_next = prob_model(row, col, action)
                            Q[row, col, action, t] = R[row, col] + V[row_next, col_next, t + 1]
                        ...
                    V[row, col, t] = max(Q[row, col, :, t])

    return V, Q

def findPolicy(V, Q, probModel=None):
    pi = np.zeros((Q.shape[0], Q.shape[1], Q.shape[3]))
    for t in range(Q.shape[3]):
        for row in range(Q.shape[0]):
            for col in range(Q.shape[1]):
                pi[row, col, t] = np.argmax(Q[row, col, :,t])
    return pi
