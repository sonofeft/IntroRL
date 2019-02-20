import sys
import matplotlib
import matplotlib.pyplot as plt
from introrl.td_funcs.td0_prediction import td0_prediction
from introrl.utils.running_ave import RunningAve
from introrl.mc_funcs.mc_ev_prediction import mc_every_visit_prediction
from introrl.policy import Policy
from introrl.agent_supt.state_value_coll import StateValueColl
from introrl.agent_supt.nstep_td_eval_walker import NStepTDWalker
from introrl.mdp_data.random_walk_generic_mrp import get_random_walk
from introrl.agent_supt.episode_maker import make_episode

GAMMA=1.0

AVE_OVER = 100

rw_mrp = get_random_walk(Nside_states=9, win_reward=1.0, lose_reward=-1.0, step_reward=0.0)
policy = Policy( environment=rw_mrp )

policy.intialize_policy_to_equiprobable() # should be equiprobable from above init already

episode_obj = make_episode( 'C', policy, rw_mrp )

fig, ax = plt.subplots()

# ---------------- set up true value data for RMS calc --------------------
true_valueD = {'C':0.0} # { 'Win':0.0, 'Lose':0.0}

#print('rw_mrp.get_num_states() = ',rw_mrp.get_num_states())
delta = 2.0 / (rw_mrp.get_num_states()-1)
Nsides = int( rw_mrp.get_num_states() / 2) - 1
d = 0.0
for i in range(1, Nsides+1 ):
    d += delta
    true_valueD[ 'L-%i'%i] = float('%g'%-d) # I got mad about the small bits.
    true_valueD[ 'R+%i'%i] = float('%g'%d)

#print('true_valueD =',true_valueD)
#sys.exit()

# ----------------------------------------- generate data -------------
alphaL = [0.05*n for n in range(21)]
nstepL = [1,2,4,8, 16, 32]

nstep_walkerL = []
ave_rms_aveD = {} # index=(alpha, Nsteps), value=RunningAve
sv_collD = {} # index=(alpha, Nsteps), value=StateValueColl

# create data structures
for Nsteps in nstepL:
    nstep_walkerL.append( NStepTDWalker(rw_mrp, Nsteps=Nsteps,  episode_obj=episode_obj) )
    for alpha in alphaL:
        ave_rms_aveD[ (alpha, Nsteps) ] = RunningAve()
        sv_collD[ (alpha, Nsteps) ] = StateValueColl( rw_mrp, init_val=0.0 )

# begin main loop over runs
for loop in range(AVE_OVER): # average rms curves over AVE_OVER runs
    if loop%10==0:
        print(loop, end='')
    else:
        print('.', end='')
    
    # set state variables to 0.0
    for Nsteps in  nstepL :
        for  alpha in alphaL:
            sv_collD[ (alpha, Nsteps) ].init_Vs_to_val( 0.0 )

            # get the initial RMS 
            #rms = sv_collD[ (alpha, Nsteps) ].calc_rms_error( true_valueD )
            #ave_rms_aveD[ (alpha, Nsteps) ].add_val( rms )


    # look at first 10 episodes
    for _ in range(10):
        episode_obj = make_episode( 'C', policy, rw_mrp )
        
        # have each alpha and Nsteps use the same episode_obj
        for istep,Nsteps in enumerate( nstepL ):
            nstep_walkerL[ istep ].set_episode_obj( episode_obj )
            
            for alpha in alphaL:
                nstep_walkerL[ istep ].do_td_state_value_updates( sv_collD[ (alpha, Nsteps) ], alpha=alpha, gamma=GAMMA)

                # get the RMS after each episode.
                rms = sv_collD[ (alpha, Nsteps) ].calc_rms_error( true_valueD )
                ave_rms_aveD[ (alpha, Nsteps) ].add_val( rms )
        
print()

colorD = {1:'r', 2:'g', 4:'b', 8:'k', 16:'m', 32:'c', 64:'y'}

for Nsteps in  nstepL :
    ave_rmsL = []
    for  alpha in alphaL:
        ave_rmsL.append( ave_rms_aveD[ (alpha, Nsteps) ].get_ave() )
    ax.plot(alphaL, ave_rmsL, '%s-'%colorD[Nsteps], label='Nsteps=%i, IntroRL'%Nsteps )

    
plt.ylabel('Ave. RMS Error (100 experiments)')
plt.xlabel('Alpha ' + r'($\alpha$)')

# Digitized Sutton & Barto Figure 7.2
alpha_1L = [0.000466921,0.0557666,0.107186,0.167336,0.259502,0.364281,0.489432,0.608763,0.697049,0.740706,0.82026,0.906605,0.998771]
rms_1L = [0.547304,0.523969,0.504338,0.482114,0.452483,0.42211,0.390997,0.366922,0.354698,0.350254,0.347291,0.353587,0.390256]
ax.plot(alpha_1L, rms_1L, '%s:'%colorD[1], label='Nsteps=1, Sutton' )

alpha_2L = [0,0.0630475,0.112517,0.183867,0.278049,0.385549,0.474023,0.547276,0.60721,0.675706,0.752764,0.826017,0.903075,0.996305]
rms_2L = [0.546836,0.497455,0.460782,0.415758,0.364198,0.317721,0.291578,0.281411,0.281411,0.288673,0.307554,0.331156,0.367466,0.437544]
ax.plot(alpha_2L, rms_2L, '%s:'%colorD[2], label='Nsteps=2, Sutton' )

alpha_4L = [0.000259455,0.0440208,0.0820741,0.130592,0.19338,0.267584,0.330372,0.370328,0.429311,0.519687,0.638604,0.774645,0.904026,0.999159]
rms_4L = [0.54611,0.480753,0.43355,0.381264,0.328251,0.286858,0.270155,0.267614,0.272334,0.290852,0.328251,0.383442,0.449526,0.515247]
ax.plot(alpha_4L, rms_4L, '%s:'%colorD[4], label='Nsteps=4, Sutton' )

alpha_8L = [-0.000691879,0.0192861,0.0421181,0.0725608,0.108711,0.15057,0.190526,0.220969,0.274243,0.373182,0.525395,0.672852,0.814601,0.931615]
rms_8L = [0.547926,0.495276,0.442627,0.385984,0.336239,0.300656,0.284316,0.280685,0.285769,0.31409,0.372912,0.433187,0.492735,0.549741]
ax.plot(alpha_8L, rms_8L, '%s:'%colorD[8], label='Nsteps=8, Sutton' )

alpha_16L = [-0.000503249,0.0198703,0.0431544,0.0615876,0.0839015,0.109126,0.135321,0.181889,0.240099,0.308981,0.386594,0.459357,0.560255,0.721303]
rms_16L = [0.546934,0.460631,0.396183,0.360254,0.333586,0.317659,0.312844,0.322104,0.345439,0.379515,0.416554,0.449149,0.489893,0.549526]
ax.plot(alpha_16L, rms_16L, '%s:'%colorD[16], label='Nsteps=16, Sutton' )

alpha_32L = [-0.000471227,0.0101724,0.0231813,0.0385555,0.0557036,0.0698951,0.0829041,0.0994609,0.116609,0.154453,0.199984,0.260298,0.331256,0.383883,0.433553]
rms_32L = [0.548029,0.48583,0.434262,0.395134,0.370707,0.362564,0.360981,0.363695,0.371385,0.394003,0.424989,0.463665,0.504151,0.528804,0.550064]
ax.plot(alpha_32L, rms_32L, '%s:'%colorD[32], label='Nsteps=32, Sutton' )


ax.legend( fontsize=8, loc='upper left', bbox_to_anchor=(0.15, 0.99) )
ax.set(title='Figure 7.2 Gamma = %g, Ave Over %i Runs\n'%(GAMMA, AVE_OVER) +\
             rw_mrp.info + ' with Equiprobable Policy' )
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')
fig.savefig("figure_7_2_random_walk_19.png")
plt.show()
