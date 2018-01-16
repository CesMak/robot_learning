def prob_model(row, col, action, use_prob_model=False):
    if not use_prob_model:
        ...
    elif use_prob_model:
        row_next = np.zeros(4)
        col_next = np.zeros(4)
        prob = np.zeros(4)
        if action == 0:
            # down
            row_next[0], col_next[0] = prob_model(row, col, 0)
            prob[0] = np.array([0.7])
            row_next[1], col_next[1] = prob_model(row, col, 1) # right
            prob[1] = 0.1
            row_next[2], col_next[2] = prob_model(row, col, 3) # left
            prob[2] = 0.1
            row_next[3], col_next[3] = prob_model(row, col, 4) # stay
            prob[3] = 0.1
        elif action == 1:
            # right
            row_next[0], col_next[0] = prob_model(row, col, 1)
            prob[0] = np.array([0.7])
            row_next[1], col_next[1] = prob_model(row, col, 0)  # down
            prob[1] = 0.1
            row_next[2], col_next[2] = prob_model(row, col, 2)  # up
            prob[2] = 0.1
            row_next[3], col_next[3] = prob_model(row, col, 4)  # stay
            prob[3] = 0.1
        elif action == 2:
            # up
            row_next[0], col_next[0] = prob_model(row, col, 2)
            prob[0] = np.array([0.7])
            row_next[1], col_next[1] = prob_model(row, col, 1)  # right
            prob[1] = 0.1
            row_next[2], col_next[2] = prob_model(row, col, 3)  # left
            prob[2] = 0.1
            row_next[3], col_next[3] = prob_model(row, col, 4)  # stay
            prob[3] = 0.1
        elif action == 3:
            # left
            row_next[0], col_next[0] = prob_model(row, col, 3)
            prob[0] = np.array([0.7])
            row_next[1], col_next[1] = prob_model(row, col, 0)  # down
            prob[1] = 0.1
            row_next[2], col_next[2] = prob_model(row, col, 2)  # up
            prob[2] = 0.1
            row_next[3], col_next[3] = prob_model(row, col, 4)  # stay
            prob[3] = 0.1
        elif action == 4:
            # prep
            row_next = np.array([row])
            col_next = np.array([col])
            prob = np.array([1])
        return row_next, col_next, prob

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
        # iterate backwards in time! for schleife rückwärts
        for t in range(T - 2, -1, -1):
            # iterate over all state rows
            for row in range(R.shape[0]):
                # iterate over all columns
                for col in range(R.shape[1]):
                    # iterate over all actions
                    for action in range(actions.shape[0]):
                        if not probModel:
                            ...
                        elif probModel:
                            #Task 2.d) : each next state or action has additionally a probability value
                            row_next, col_next, prob = prob_model(row, col, action, use_prob_model=True)
                            Q[row, col, action, t] = R[row, col]
                            for i in range(row_next.shape[0]):
                                Q[row, col, action, t] = Q[row, col, action, t]+prob[i] * V[int(row_next[i]), int(col_next[i]), t + 1]
                    V[row, col, t] = max(Q[row, col, :, t])
    return V, Q
