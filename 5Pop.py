'''
"Paradoxical response reversal of top-down modulation in cortical circuits with three interneuron types."
Luis Carlos Garcia, Guangyu Robert Yang, Jorge F. Mejias, and Xiao-Jing Wang
eLife 6 (2017): e29742
Created on 2018/4/11 by Warren Woodrich Pettine and Sean Froudist-Walsh

Modified by John to include T-shape SST and other SST in layer 5. 
Calibrate the baseline activity by Yu 2019.

01302020: Updates the code to check whether E connection is needed for competition between two E population
'''

# Import statements
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from numpy.linalg import inv

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


# Define Parameters

P = dict(
    T           =   2000,                            # Time duration for simulation (seconds)
    dt          =   0.1,                            # Time step of simulation
    r           =   np.array([1, 10, 3,3, 2]),        # initial firing rate (Pyr, PV, T-SST, nT-SST, VIP)
    gl          =   np.array([6.25,10,5,5,5]),        # Leak conductance in nS (Pyr, PV, T-SST, nT-SST, VIP)
    Cm          =   np.array([180, 80, 80,80, 80]),    # Capacitance in pF (Pyr, PV, T-SST, nT-SST, VIP)
    noise       =   1*np.array([1,1,1,1,1]),          # noise in mV
    I_mod       =   1,
#    S           =   np.array(  [[1, 1, 1/2, 1/2,0],     # Original connectivity 
#                                [1, 1, 1/2, 1/2,0],     #  Multified Connectivity matrix to indicate the population density       
#                                [1, 0, 0, 0,    1],     # Input to T-SST
#                                [1, 0, 0, 0,    1],     # Input to nT-SST       
#                                [1, 0, 1/2, 1/2,0],]),
#    S           =   np.array(  [[1, 1, 1/2, 1/2,0],     # for ~ lambda=0.6
#                                [1, 1, 1/2, 1/2,0],            
#                                [1, 0, 0,    0,    1],     # Input to T-SST
#                                [1, 0, 0,    0,    0.6],     # Input to nT-SST       
#                                [1, 0, 0.625, 0.375,    0],]), # Input to VIP
    S           =   np.array(  [[1, 1, 1/2, 1/2,0],     #  No E-S connection
                                [1, 1, 1/2, 1/2,0],            
                                [1, 0, 0,    0,    1],     # Input to T-SST
                                [1, 0, 0,    0,    0.6],     # Input to nT-SST       
                                [1, 0, 0.625, 0.375,    0],]), # Input to VIP



    W           =   np.array([  [2.4167, -0.3329, -0.8039,-0.8039,  0], # Published weight values in pAs
                                [2.971,  -3.455,  -2.129 ,-2.129,  0],
                                [4.644,  0,       0,      0,     -2.7896],
                                [4.644,  0,       0,      0,     -2.7896],
                                [0.7162, 0,       -0.1560, -0.1560,   0]]),
    V_l         =   -70,                            # Leak
    V_th        =   -50,                            # Threshold
    V_reset     =   -60,                            # Reset potential
    rtau      =   np.array( [[ 1], [1], [1],  [1],[1]   ]             ),
    lamb =  0.6
)

# Create simulation class
class Simululation(object):
    def __init__(self,P):
        self.P = P
        lamb= self.P['lamb']
        self.P['S'][4,2]=1/(1+lamb);
        self.P['S'][4,3]=lamb/(1+lamb);
        self.P['S'][2,4]=2/(1+lamb);
        self.P['S'][3,4]=2*lamb/(1+lamb);
        
        # Ensure dimensions are correct
        try:
            self.P['r'].shape = (5, 1)
            self.P['gl'].shape = (5, 1)
            self.P['Cm'].shape = (5, 1)
            self.P['noise'].shape = (5, 1)
        except ValueError:
            print('Four values should be given for parameters: r0, r1, gl, Cm, noise')

        # membrane time constant in seconds (Pyr, PV, SST, VIP)
        self.P['tau'] = np.ndarray(buffer=self.P['Cm']/self.P['gl']/1000,shape=(5,1))

        # Steady state voltages
        self.P['V_base'] = np.ndarray(buffer=self.optmiz(self.P['r']), shape=(5,1))

        # Variables for the simulation
        self.I_stim = np.zeros((5, 1))                                  # Input current
        self.I_stim[3] = 0
        self.I_bkg = (self.P['V_base'] - self.P['V_l']) * self.P['gl'] - np.dot(self.P['W']*self.P['S'], self.P['r'])  # Background current
        self.r_history = np.zeros((5,int((self.P['T']+self.P['dt'])*10)))          # Firing rate history
        self.time = np.linspace(self.P['dt'], self.P['T'], int(self.P['T']/self.P['dt']) + 1)

    # f-I curve from Abbott & Chance, 2005
    def phi(self,V_i):
        return (V_i - self.P['V_th']) / (self.P['tau'] * (self.P['V_th'] -
                self.P['V_reset']) * (1 - np.exp(-(V_i - self.P['V_th']) / self.P['noise'])))

    # Derivative of f-I curve
    def dphi(self,V_i):
        return  (1 - (V_i - self.P['V_th']) * np.exp(-(V_i - self.P['V_th']) / self.P['noise']) / self.P['noise']) / \
                (self.P['tau']  * (self.P['V_th'] - self.P['V_reset']) * (1 - np.exp(-(V_i - self.P['V_th']) / self.P['noise'])) ** 2)

    # Find voltage steady state that leads to spontaneous baseline firing rate
    def optmiz(self,r):
        return scipy.optimize.newton_krylov(lambda V: r - self.phi(V), np.zeros(r.shape))

    # Run the simulation
    def runSim(self):
        r = self.P['r']
        for t in range(int((self.P['T']+self.P['dt'])*10)):
            
            # Turn on current to VIP neurons
            if self.time[t] > self.P['T'] / 4:
                self.I_stim[0] = self.P['I_mod']
                self.I_stim[1] = 2*self.P['I_mod']
                self.I_stim[4] = 4*self.P['I_mod']   # Activated by non-sensory inputs, by Schuman, Rudy 2019.
#            self.I_stim[4] =-self.P['I_mod']

            # SFW: equation 2 from the Results section
            V = self.P['V_l'] + (   (self.P['W']*self.P['S']).dot(r) + self.I_stim + self.I_bkg) / self.P['gl']

            # Calculate the update in firing rates
            dr = self.P['dt'] * (-r + self.phi(V)) / self.P['rtau'] 

            # perform update of firing rates for all populations
            r = r + dr

            # update history of firing rates
            self.r_history[:, t] = r.flatten()

    def plotphi(self):
        colours = np.array([[8, 48, 107],  # dark-blue
                            [228, 26, 28],  # red
                            [152, 78, 163],  # purple
                            [0, 0, 0],  # purple
                            [77, 175, 74]]) / 255.  # green
        plt.figure()
        Vaxis=  np.linspace(-60,-40,200)
        Rcalmat= np.zeros((5,200))
        for iv in range(200):
            Vol=Vaxis[iv]*np.ones((5, 1))      
            temp=self.phi(Vol);
            Rcalmat[:,iv]= temp.flatten()
            
        for p in range(5):
            plt.plot(Vaxis, Rcalmat[p, :], color=colours[p, :])    
            
        xmin = -60
        xmax = -40
        ymin = 0
     #   ymax = np.ceil(self.r_history.max()) + 8
        ymax= 50
        
        axes = plt.gca()
        axes.set_xlim([xmin, xmax])
        axes.set_ylim([ymin, ymax])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        plt.xlabel('$\it{V}$ (mV)', fontsize=12)
        plt.ylabel('$\it{r}$ (Hz)', fontsize=12)
        plt.title("Activation Function", fontsize=16)
        plt.legend(['Pyr', 'PV', 'T-SST','nT-SST', 'VIP'], frameon=False)
        
        plt.show
        
        
    def CalRevM(self):
        temp=self.dphi(self.P['V_base'])
        matD=np.eye(5)*(self.P['gl']/ temp)
        matW=self.P['W']*self.P['S']
        matM= inv(matD-matW)
        self.matM=matM
        print(np.linalg.det(matD-matW))
        fig,ax = plt.subplots()
        im=ax.imshow(matM)
        for i in range(5):  # CHANGED I TO E, NEED TO CHECK IF RIGHT
           for j in range(5):
               plt.text(i, j, np.round(matM[j, i], 2), ha="center", va="center", color="w", fontsize=12)
        
        
        divider1 =make_axes_locatable(ax)
        cax1=divider1.append_axes("right",size="7%",pad="2%")
        cb=colorbar(im,cax=cax1)
        Cell_Type=['E','PV','T-SST','nT-SST','VIP']
        x_pos=np.arange(len(Cell_Type))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(Cell_Type)
        y_pos=np.arange(len(Cell_Type))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(Cell_Type)        
        ax.set_ylim([-0.5, 4.5])
        tt1='Response Mat lamb=%.1f' % self.P['lamb']
#        tt1='Response Mat Sanity'
        ax.set_title(tt1)#        cb.set_clim(vmin=-2, vmax=4)
#        plt.savefig(tt1.replace(" ", "_") + '.pdf')
        plt.show()
 
    # Plot the results
    def plotFR(self, ttl='Population FRs', sv_fig=True):
        # Define colours
        colours = np.array([[8, 48, 107],  # dark-blue
                            [228, 26, 28],  # red
                            [152, 78, 163],  # purple
                            [0, 0, 0],  # purple
                            [77, 175, 74]]) / 255.  # green

        # Create a new figure
        plt.figure()

        # Time X-axis
        time = self.time - self.P['T'] / 4 + 5

        # Set the bar height for top-down modulation
        top = np.ceil(self.r_history.max()) + 2

        # Plot solution
        for p in range(5):
            plt.plot(time, self.r_history[p, :], color=colours[p, :])
        plt.plot([5, self.P['T']], [top, top], color='black', linewidth=5.0)
        xmin = -10
        xmax = 40
        ymin = 0
     #   ymax = np.ceil(self.r_history.max()) + 8
        ymax=40
        
        axes = plt.gca()
        axes.set_xlim([xmin, xmax])
        axes.set_ylim([ymin, ymax])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        plt.xlabel('$\it{t}$ (ms)', fontsize=12)
        plt.ylabel('$\it{r}$ (Hz)', fontsize=12)
        plt.title(ttl, fontsize=16)
        plt.legend(['Pyr', 'PV', 'T-SST','nT-SST', 'VIP'], frameon=False)

        # place label above the black line
        axes.text(0.45, 1, 'Whisking', transform=axes.transAxes, fontsize=10, verticalalignment='top')

        # save and close the figure
        if sv_fig:
            plt.savefig(ttl.replace(" ", "_") + '.pdf')
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    # Run simulation for low baseline condition
#   P['r'] = np.array([1, 10, 3, 3, 2])  # initial firing rate for low firing rate condition (Pyr, PV, SST, VIP)
#    P['r'] = np.array([7, 8, 3, 3, 17])   # Initial rate based on Yu, Sovoboda 2019 in layer 5
#    P['r'] = np.array([1, 10, 8, 8, 5])   # Initial rate based on Gentet, Peterson 2012 in layer 2/3
    P['r'] = np.array([1, 8, 3, 3, 17])
    sim = Simululation(P)
#    sim.plotphi()
    sim.CalRevM()
    sim.runSim()
    sim.plotFR('T-SST vs. nT-ssT noVIP',False)
'''

    # Run Simulation for high baseline condition
   # P['r'] = np.array([1, 10, 3, 2])  # initial firing rate for low firing rate condition (Pyr, PV, SST, VIP)

    P['r'] = np.array([30, 50, 30,30, 20])  # initial firing rate for high firing rate condition (Pyr, PV, SST, VIP)
    sim = Simululation(P)
    sim.runSim()
#    sim.plotFR('Control',False)
    sim.plotFR('VIP No Change ',True)
'''
#    sim.plotFR('Sanity',True)