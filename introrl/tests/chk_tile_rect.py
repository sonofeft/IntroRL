import matplotlib.pyplot as plt
import random
from introrl.utils.tiles_rectangles import Tile, Tilings, plot_tile

t = Tile(lo_valL=[0,1], hi_valL=[1,10], num_regionsL=[4,5])
T = Tilings( t, num_tiles=4, recenter=True, show_pow2_warning=True)

fig, ax = plt.subplots()

n=0
colorL = ['r','g','b','c','m','y']
linestyleL = ['-',':','--']
for itile, tile in enumerate(T.tileL):
    plot_tile(ax, tile, color=colorL[itile%len(colorL)], 
              linestyle=linestyleL[itile%len(linestyleL)], n=n )

for _ in range(10):
    x = -0.2 + random.random()*1.2
    y = random.random() * 11
    plt.plot(x,y,'rs')
    s = str( T.get_regions( [x,y] ) )
    plt.xlim(-.2, 1.4)
    plt.text(x,y,s)

plt.show()

