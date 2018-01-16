
def ValIter(R, discount=None, maxSteps=None, infHor=False, probModel=False):
    #R = grid_world matrix! state metric for reward
    ...
    else:
        # infinite horizon:
        # down, right, up, left, prep
        actions = np.array([0, 1, 2, 3, 4])
        max_k = maxSteps
        Q = np.zeros((R.shape[0], R.shape[1], actions.shape[0], max_k))
        V = np.zeros((R.shape[0], R.shape[1], max_k))
        k = 0
        # for schleife vorwärts:
        for k in range(max_k-1):
            # iterate over all state rows
            for row in range(R.shape[0]):
                # iterate over all columns
                for col in range(R.shape[1]):
                    # iterate over all actions
                    for action in range(actions.shape[0]):
                        row_next, col_next = prob_model(row, col, action)
                        Q[row, col, action, k+1] = R[row, col] + discount*V[row_next, col_next, k]
                    V[row, col, k+1] = max(Q[row, col, :, k+1])

            #c heck for convergence
            if (np.abs(V[:,:,k]-V[:,:,k+1])<0.01).all():
                V = V[:,:,0:k+1]
                Q = Q[:,:,:,0:k+1]
                break
        V = np.flip(V, axis=2)
        Q = np.flip(Q, axis=3)

    return V, Q
