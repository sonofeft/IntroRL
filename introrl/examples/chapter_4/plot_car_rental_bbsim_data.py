from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from introrl.dp_funcs.dp_policy_iter import dp_policy_iteration
from introrl.policy import Policy
from introrl.state_values import StateValues
from introrl.utils import pickle_esp

env, state_value, policy = pickle_esp.read_pickle_file( fname='car_rental_sim_to_env_const_rtn' )

# saved file ran through value iteration, for comparison, run through policy iteration
dp_policy_iteration(policy, state_value, 
                    do_summ_print=True, show_start_policy=True,
                    max_iter=1000, err_delta=0.0001, gamma=0.9)


diag_colorD = {'5':'r', '4':'g', '3':'b', '2':'c', '1':'y', '0':'w', 
               '-5':'r', '-4':'g', '-3':'b', '-2':'c', '-1':'y'}
                   
policy.save_diagram( env, inp_colorD=diag_colorD, save_name='policy_car_rental_sim_to_env_const_rtn',
                     show_arrows=False, scale=0.25, h_over_w=0.8, do_show=False)

state_value.summ_print( fmt_V='%.1f')


# --------------------------------------------------------------
fig = plt.figure( figsize=(8,6) )
ax = fig.gca(projection='3d')

# Make data.
X = list( range(0, 21) )
Y = list( range(0, 21) )
Z = []
for y in  Y:
    rowL = []
    for x in X:
        s_hash = (x, y)
        rowL.append( state_value(s_hash) )
    Z.append( rowL )

X, Y = np.meshgrid(X, Y)
Z = np.array( Z )

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(420, 620)

ax.set_ylim(0, 20)
ax.set_xlim(0, 20)

plt.xticks( [0, 20],['0','20'] )
plt.yticks( [0, 20],['0','20'] )

ax.set_zticks( [420,612] )

ax.set_title( "Jack's Car Rental State Values" )
ax.view_init( elev=45.0, azim=-65.0)

ax.set_xlabel('#Cars at second location')
ax.set_ylabel('#Cars at first location')
ax.set_zlabel('V(s)')

fig.savefig("car_rental_sim_to_env_const_rtn.png")


# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

