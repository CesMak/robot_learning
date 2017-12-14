import numpy as np
import matplotlib.pyplot as plt
from getImitationData import *
from getProMPBasis import *
from scipy.linalg import inv

def proMP (nBasis, condition=False):

    dt = 0.002
    time = np.arange(dt,3,dt)
    nSteps = len(time)
    data = getImitationData(dt, time, multiple_demos=True)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    bandwidth = 0.2
    Phi = getProMPBasis( dt, nSteps, nBasis, bandwidth )

    # estimate weight vectors w for all demonstrations
    W = np.eye(Phi.shape[1])*1e-7
    pseudo_inv = inv(np.dot(Phi.T, Phi) + W)
    ws = np.dot(np.dot(pseudo_inv, Phi.T), q.T)

    # fit a Gaussian over all weight vectors
    
    mean_w = np.mean(ws, axis=1)
    cov_w = np.cov(ws)
    mean_traj = np.dot(Phi, mean_w)
    std_traj = np.diag(np.dot(np.dot(Phi, cov_w), Phi.T))

    plt.figure()
    plt.hold('on')
    plt.plot(time, q.T, color=[0.5,0.5,0.5], alpha=0.3)
    for i in xrange(q.shape[0]):
        y = np.dot(Phi, ws[:,i])
        plt.plot(time, y, color='orange', alpha=0.3)

    plt.fill_between(time, mean_traj - 2*std_traj, mean_traj + 2*std_traj, alpha=1, edgecolor='#1B2ACC', facecolor='#089FFF')

    plt.plot(time, mean_traj, color='#1B2ACC')

    plt.title('ProMP with ' + str(nBasis) + ' basis functions')

    Phi = Phi.T
    #Conditioning
    if condition:
        y_d = 3
        Sig_d = 0.0002
        t_point = np.round(2300/2)

        tmp = np.dot(cov_w, Phi[:,t_point]) / (Sig_d + np.dot(Phi[:,t_point].T,np.dot(cov_w,Phi[:,t_point])))

        cov_w_new = 0#...
        mean_w_new = 0#...
        mean_traj_new = 0#...
        std_traj_new = 0#...

        plt.figure()
        plt.hold('on')
        plt.fill_between(time, mean_traj - 2*std_traj, mean_traj + 2*std_traj, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
        plt.plot(time,mean_traj, color='#1B2ACC')
        plt.fill_between(time, mean_traj_new - 2*std_traj_new, mean_traj_new + 2*std_traj_new, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.plot(time, mean_traj_new, color='#CC4F1B')

        sample_traj = np.dot(Phi.T,np.random.multivariate_normal(mean_w_new,cov_w_new,10).T)
        plt.plot(time,sample_traj)
        plt.title('ProMP after contidioning with new sampled trajectories')

if __name__=="__main__":
    proMP(30)
    plt.show()
