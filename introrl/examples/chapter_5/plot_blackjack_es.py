import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from introrl.black_box_sims.blackjack_sim import BlackJackSimulation
from introrl.policy import Policy
from introrl.agent_supt.action_value_run_ave_coll import ActionValueRunAveColl

BJ = BlackJackSimulation()
av = ActionValueRunAveColl( BJ )
av.init_from_pickle_file( fname='blackjack_es')

sv = av.build_Vs_from_Qsa( BJ )
sv.summ_print( showRunningAve=False )


# --------------------------------------------------------------
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = list( range(12, 22) )
Y = list( range(1, 11) )
Z = []
for y in  Y:
    rowL = []
    for x in X:
        s_hash = (x, True, y)
        rowL.append( sv.get_ave(s_hash) )
    Z.append( rowL )

X, Y = np.meshgrid(X, Y)
Z = np.array( Z )

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_ylim(10, 1)
ax.set_xlim(12, 21)
ax.set_title( 'Usable Ace Monte Carlo ES' )

ax.set_xlabel('Player Sum')
ax.set_ylabel('Dealer Showing')
ax.set_zlabel('V(s)')

ax.view_init( elev=30.0, azim=-145.0)

fig.savefig("fig_5_2_w_ace_mc_es.png")

# --------------------------------------------------------------
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = list( range(12, 22) )
Y = list( range(1, 11) )
Z = []
for y in  Y:
    rowL = []
    for x in X:
        s_hash = (x, False, y)
        rowL.append( sv.get_ave(s_hash) )
    Z.append( rowL )

X, Y = np.meshgrid(X, Y)
Z = np.array( Z )

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_ylim(10, 1)
ax.set_xlim(12, 21)
ax.set_title( 'No Usable Ace  Monte Carlo ES' )

ax.set_xlabel('Player Sum')
ax.set_ylabel('Dealer Showing')
ax.set_zlabel('V(s)')

ax.view_init( elev=30.0, azim=-145.0)

fig.savefig("fig_5_2_noace_mc_es.png")


# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

