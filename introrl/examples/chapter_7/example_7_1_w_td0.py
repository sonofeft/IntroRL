import matplotlib
import matplotlib.pyplot as plt
from introrl.td_funcs.td0_prediction import td0_prediction
from introrl.utils.running_ave import RunningAve
from introrl.mc_funcs.mc_ev_prediction import mc_every_visit_prediction
from introrl.policy import Policy
from introrl.agent_supt.state_value_coll import StateValueColl
from introrl.mdp_data.random_walk_generic_mrp import get_random_walk

rw_mrp = get_random_walk(Nside_states=9, win_reward=1.0, lose_reward=-1.0, step_reward=0.0)

policy = Policy( environment=rw_mrp )

fig, ax = plt.subplots()

# ---------------- set up true value data for RMS calc --------------------
true_valueD = { 'C':0.0, 'Win':0.0, 'Lose':0.0}
    
delta = 2.0 / (rw_mrp.get_num_states() + 1)
Nsides = int( rw_mrp.get_num_states() / 2) - 1
d = 0.0
for i in range(1, Nsides+1 ):
    d += delta
    true_valueD = { 'L-%i'%i:-d}
    true_valueD = { 'R+%i'%i:d}

# ----------------------------------------- generate TD(0) data -------------
alphaL = [0.01] + [0.05*n for n in range(1,21)]
ave_rms_aveL = [RunningAve(name='alpha=%g'%alpha) for alpha in alphaL]

for ialpha, alpha in enumerate(alphaL):

    for loop in range(100): # average rms curves over 100 runs
        sv = StateValueColl( rw_mrp, init_val=0.5 )
        
        resultL, value_snapD = td0_prediction( policy, sv,  all_start_states=False,
                               do_summ_print=False, show_last_change=False,
                               show_banner = False,
                               pcent_progress_print=0,
                               alpha=alpha, const_alpha=True, alpha_half_life=200,
                               max_episode_steps=100000,
                               max_num_episodes=10, min_num_episodes=10, max_abserr=0.001, gamma=1.0,
                               result_list='rms', true_valueD=true_valueD,
                               value_snapshot_loopL=None) # if input, save V(s) snapshot at iteration steps indicated
                               
        rms_ave = sum(resultL) / len(resultL)
        ave_rms_aveL[ialpha].add_val( rms_ave )
        
    if ialpha%10==0:
        print('#', end='')
    else:
        print('.', end='')
print()
ave_rmsL = [R.get_ave() for R in ave_rms_aveL]
ax.plot(alphaL, ave_rmsL, 'c', label='TD(0)')
    
plt.ylabel('Ave. RMS Error (100 experiments)')
plt.xlabel('Alpha')
    
    
ax.legend()
ax.set(title='Example 7.1 with TD(0) '+ rw_mrp.info )
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')
fig.savefig("example_7.1_with_td0_random_walk_19.png")
plt.show()
