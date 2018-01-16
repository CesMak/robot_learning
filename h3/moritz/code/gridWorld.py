import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


##
def genGridWorld():
    O = -1e5  # Dangerous places to avoid
    D = 35    # Dirt
    W = -100  # Water
    C = -3000 # Cat
    T = 1000  # Toy
    grid_list = {0:'', O:'O', D:'D', W:'W', C:'C', T:'T'}
    grid_world = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
        [0, 0, 0, 0, D, O, 0, 0, D, 0],
        [0, D, 0, 0, 0, O, 0, 0, O, 0],
        [O, O, O, O, 0, O, 0, O, O, O],
        [D, 0, 0, D, 0, O, T, D, 0, 0],
        [0, O, D, D, 0, O, W, 0, 0, 0],
        [W, O, 0, O, 0, O, D, O, O, 0],
        [W, 0, 0, O, D, 0, 0, O, D, 0],
        [0, 0, 0, D, C, O, 0, 0, D, 0]])
    return grid_world, grid_list


##
def showWorld(grid_world, tlt):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(tlt)
    ax.set_xticks(np.arange(0.5,10.5,1))
    ax.set_yticks(np.arange(0.5,9.5,1))
    ax.grid(color='b', linestyle='-', linewidth=1)
    ax.imshow(grid_world, interpolation='nearest', cmap='copper')
    return ax


##
def showTextState(grid_world, grid_list, ax):
    for x in xrange(grid_world.shape[0]):
        for y in xrange(grid_world.shape[1]):
            if grid_world[x,y] >= -3000:
                ax.annotate(grid_list.get(grid_world[x,y]), xy=(y,x), horizontalalignment='center')


##
def showPolicy(policy, ax):
    for x in xrange(policy.shape[0]):
        for y in xrange(policy.shape[1]):
            if policy[x,y] == 0:
                ax.annotate('$\downarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 1:
                ax.annotate(r'$\rightarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 2:
                ax.annotate('$\uparrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 3:
                ax.annotate('$\leftarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 4:
                ax.annotate('$\perp$', xy=(y,x), horizontalalignment='center')


##
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
            # prep
            row_next = row
            col_next = col
        return row_next, col_next
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
        for t in xrange(T - 2, -1, -1):
            # iterate over all state rows
            for row in xrange(R.shape[0]):
                # iterate over all columns
                for col in xrange(R.shape[1]):
                    # iterate over all actions
                    for action in xrange(actions.shape[0]):
                        if not probModel:
                            row_next, col_next = prob_model(row, col, action)
                            Q[row, col, action, t] = R[row, col] + V[row_next, col_next, t + 1]
                        elif probModel:
                            row_next, col_next, prob = prob_model(row, col, action, use_prob_model=True)
                            Q[row, col, action, t] = R[row, col]
                            for i in xrange(row_next.shape[0]):
                                Q[row, col, action, t] += prob[i] * V[int(row_next[i]), int(col_next[i]), t + 1]

                    V[row, col, t] = max(Q[row, col, :, t])
    else:
        # down, right, up, left, prep
        actions = np.array([0, 1, 2, 3, 4])
        max_k = maxSteps
        Q = np.zeros((R.shape[0], R.shape[1], actions.shape[0], max_k))
        V = np.zeros((R.shape[0], R.shape[1], max_k))
        pi = np.zeros((Q.shape[0], Q.shape[1], Q.shape[3]))
        k = 0
        for k in xrange(max_k-1):
            # iterate over all state rows
            for row in xrange(R.shape[0]):
                # iterate over all columns
                for col in xrange(R.shape[1]):
                    # iterate over all actions
                    for action in xrange(actions.shape[0]):
                        row_next, col_next = prob_model(row, col, action)
                        Q[row, col, action, k+1] = R[row, col] + discount*V[row_next, col_next, k]

                    for action in xrange(actions.shape[0]):
                        pi[row, col, k+1] = np.argmax(Q[row, col, :, k+1])
                        V[row, col, k+1] += pi[row, col, k+1] * Q[row, col, action, k+1]
                    V[row, col, k+1] = max(Q[row, col, :, k+1])

            if (np.abs(V[:,:,k]-V[:,:,k+1])<0.01).all():
                V = V[:,:,0:k+1]
                Q = Q[:,:,:,0:k+1]
                break
        V = np.flip(V, axis=2)
        Q = np.flip(Q, axis=3)

    return V, Q


##
def maxAction(V, R, discount, probModel=None):
    return 0

##
def findPolicy(V, Q, probModel=None):
    pi = np.zeros((Q.shape[0], Q.shape[1], Q.shape[3]))
    for t in xrange(Q.shape[3]):
        for row in xrange(Q.shape[0]):
            for col in xrange(Q.shape[1]):
                pi[row, col, t] = np.argmax(Q[row, col, :,t])
    return pi

############################

saveFigures = False

data = genGridWorld()
grid_world = data[0]
grid_list = data[1]

#probModel = ...

ax = showWorld(grid_world, 'Environment')
showTextState(grid_world, grid_list, ax)
if saveFigures:
    plt.savefig('gridworld.pdf')

# Finite Horizon
V, Q = ValIter(grid_world, maxSteps=15)
V = V[:,:,0];
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon')
if saveFigures:
    plt.savefig('value_Fin_15.pdf')

policy = findPolicy(V, Q)
ax = showWorld(grid_world, 'Policy - Finite Horizon')
showPolicy(policy[:,:,0], ax)
if saveFigures:
    plt.savefig('policy_Fin_15.pdf')


#for i in [9,10,11]:
#    V, Q = ValIter(grid_world, maxSteps=i)
#    V = V[:, :, 0];
#    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon steps {}'.format(i))
#    if saveFigures:
#        plt.savefig('value_Fin_15_{}.pdf'.format(i))
#    policy = findPolicy(V, Q)
#    ax = showWorld(grid_world, 'Policy - Finite Horizon steps {}'.format(i))
#    showPolicy(policy[:, :, 0], ax)

# Infinite Horizon
V, Q = ValIter(grid_world, maxSteps=100, discount=0.8, infHor=True)
V = V[:,:,0];
showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
if saveFigures:
    plt.savefig('value_Inf_08.pdf')

policy = findPolicy(V, Q);
ax = showWorld(grid_world, 'Policy - Infinite Horizon')
showPolicy(policy[:,:,0], ax)
if saveFigures:
    plt.savefig('policy_Inf_08.pdf')

# Finite Horizon with Probabilistic Transition
V, Q = ValIter(grid_world, maxSteps=15, probModel=True)
V = V[:,:,0];
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
if saveFigures:
    plt.savefig('value_Fin_15_prob.pdf')

policy = findPolicy(V, Q)
ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
showPolicy(policy[:,:,0], ax)
if saveFigures:
    plt.savefig('policy_Fin_15_prob.pdf')

plt.show()
