from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from introrl.utils import pickle_esp

env, state_value, policy = pickle_esp.read_pickle_file( fname='dp_car_rental_PI' )


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

fig.savefig("fig_4_2_car_rental_value_v3.png")


# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

