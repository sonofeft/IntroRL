from introrl.policy import Policy
from introrl.black_box_sims.racetrack_1_sim import RaceTrack_1
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RT = RaceTrack_1()
#pi = Policy( environment=RT )

policyD = Policy().read_pickle_file( 'racetrack_1_sim' )


#pi.init_from_pickle_file( 'racetrack_1_sim' )

fig, ax = plt.subplots()

for (j,i) in RT.racetrack_area:
    rect = mpatches.Rectangle((i-.5, j-.5), 1.0, 1.0, ec="none", color='blue', alpha=0.3)
    ax.add_patch(rect)

for (j,i,_,_) in RT.starting_lineL:
    rect = mpatches.Rectangle((i-.5, j-.5), 1.0, 1.0, ec="none", color='yellow', alpha=1.)
    ax.add_patch(rect)

for (j,i) in RT.finish_lineL:
    rect = mpatches.Rectangle((i-.5, j-.5), 1.0, 1.0, ec="none", color='green', alpha=1.)
    ax.add_patch(rect)

markersize = len(RT.starting_lineL)*2 + 4
linewidth = len(RT.starting_lineL)*2 + 2

for start_state in RT.limited_start_state_list():
    print('For Start State =', start_state)
    
    s_hash = start_state
    (x,y, vx,vy) = s_hash
    xL=[x]
    yL=[y]
    
    
    while s_hash in policyD:
    
        a_desc = policyD[ s_hash ]
        (dvx, dvy) = a_desc
        
        x2 = x + vx
        y2 = y + vy
        vx2 = max(0, min(4, vx+dvx))
        vy2 = max(0, min(4, vy+dvy))
        
        sn_hash = (x2,y2, vx2,vy2)
        print('action ',a_desc,' leads to ',sn_hash)
        s_hash = sn_hash
        (x,y, vx,vy) = s_hash
        xL.append( x )
        yL.append( y )
    
    ax.plot(yL, xL, 'o-', markersize=markersize, linewidth=linewidth)
    markersize -= 2
    linewidth -= 2
    print('-'*55)

ax.set(title='RaceTrack_1')
ax.grid()
#plt.axis('equal')
ax.set_aspect(1.0)

fig.savefig("racetrack_1_sim.png")
plt.show()

