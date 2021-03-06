import matplotlib
import matplotlib.pyplot as plt

from introrl.td_funcs.td0_prediction import td0_prediction
from introrl.policy import Policy
from introrl.agent_supt.state_value_coll import StateValueColl
from introrl.mdp_data.random_walk_mrp import get_random_walk

rw_mrp = get_random_walk()

policy = Policy( environment=rw_mrp )


fig, ax = plt.subplots()

true_valueD = {'A':1.0/6.0, 'B':2.0/6.0, 'C':3.0/6.0, 'D':4.0/6.0, 'E':5.0/6.0}
for alpha in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15]:

    resultLL = [] # a list of result lists
    for loop in range(100): # average rms curves over 100 runs
        sv = StateValueColl( rw_mrp, init_val=0.5 )
        
        resultL, value_snapD = td0_prediction( policy, sv,  all_start_states=False,
                               do_summ_print=False, show_last_change=False,
                               show_banner=False,
                               alpha=alpha, const_alpha=True, alpha_half_life=200,
                               max_num_episodes=100, min_num_episodes=100, max_abserr=0.001, gamma=1.0,
                               result_list='rms', true_valueD=true_valueD,
                               value_snapshot_loopL=None) # if input, save V(s) snapshot at iteration steps indicated
        resultLL.append( resultL )
    #print( 'sv.calc_rms_error(true_valueD) =', sv.calc_rms_error(true_valueD) )
    #print( resultL )

    n_runs = 0
    run_numL = []
    ave_rmsL = []
    for i in range( len(resultL) ):
        n_runs += 1
        val = 0.0
        for rL in resultLL:
            val += rL[i]
        run_numL.append( n_runs )
        ave_rmsL.append( val / len(resultLL) )
        
    
    ax.plot(run_numL, ave_rmsL, label='alpha=%g'%alpha)
ax.legend()
ax.set(title='TD0 Random Walk')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
fig.savefig("td0_random_walk.png")
plt.show()
