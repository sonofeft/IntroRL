import sys
import matplotlib
import matplotlib.pyplot as plt

from introrl.mc_funcs.mc_ev_prediction import mc_every_visit_prediction
from introrl.policy import Policy
from introrl.agent_supt.state_value_coll import StateValueColl
from introrl.mdp_data.random_walk_mrp import get_random_walk
from introrl.agent_supt.episode_maker import make_episode
from introrl.utils.running_ave import RunningAve

rw_mrp = get_random_walk()
policy = Policy( environment=rw_mrp )

NumEpisodes = 100
mc_rms_raveL = [RunningAve(name='%i'%i) for i in range(NumEpisodes)]
td_rms_raveL = [RunningAve(name='%i'%i) for i in range(NumEpisodes)]

alpha = 0.1
gamma = 1.0

true_valueD = {'A':1.0/6.0, 'B':2.0/6.0, 'C':3.0/6.0, 'D':4.0/6.0, 'E':5.0/6.0}
    
def calc_td_error(show_values=True):
    errD = {} # index=s_hash, value=Vtarget - V(s)
    for s_hash in ['A','B','C','D','E']:
        errD[s_hash] = 0.0
        
    for (s_hash,sn_hash), R in td_averD.items():
        errD[s_hash] += R.get_ave() + gamma*sv_td.get_Vs(sn_hash) - sv_td.get_Vs(s_hash)
    
    if show_values:
        for (s_hash,sn_hash), R in td_averD.items():
            print( (s_hash,sn_hash), R.get_ave(), 'V(s)=%g'%sv_td.get_Vs(s_hash),
                   'V(sn)=%g'%sv_td.get_Vs(sn_hash)        ,' err=',errD[s_hash] )
        print()

    total_err = sum( [abs(err) for err in errD.values()] )
    return errD, total_err


for o_loop in range(1,101):
    print('Ex%2i'%o_loop, end=' ')
    if o_loop % 20 == 0:
        print()
            
    # make 2 state value objects.
    sv_td = StateValueColl( rw_mrp, init_val=0.5 )
    sv_mc = StateValueColl( rw_mrp, init_val=0.5 )

    mc_avegD = {} # index=s_hash, value=average G for s_hash
    td_averD = {} # index=(s_hash, sn_hash), value=average reward for s_hash

    # make 100 episodes
    for i_loop in range(NumEpisodes):
        episode = make_episode('C', policy, rw_mrp, rw_mrp.terminal_set)

        for dr in episode.get_rev_discounted_returns( gamma=gamma ):
            (s_hash, a_desc, reward, sn_hash, G) = dr
            
            if s_hash not in mc_avegD:
                mc_avegD[s_hash] = RunningAve()
            if (s_hash,sn_hash) not in td_averD:
                td_averD[(s_hash,sn_hash)] = RunningAve()
            
            mc_avegD[s_hash].add_val( G )
            td_averD[(s_hash,sn_hash)].add_val( reward )
    
        # set the Monte Carlo V(s) values for this experiment
        for s_hash, G in mc_avegD.items():
            sv_mc.set_Vs( s_hash, G.get_ave() )
    
        # set the TD(0) values for this experiment
        for update_loop in range(20):
            errD, total_err = calc_td_error( show_values=False )
            for s_hash, err in errD.items():
                sv_td.delta_update( s_hash=s_hash, delta=err*alpha)
        
        # add this loops RMS of state values to RMS running_ave
        mc_rms_raveL[i_loop].add_val( sv_mc.calc_rms_error( true_valueD ) )
        td_rms_raveL[i_loop].add_val( sv_td.calc_rms_error( true_valueD ) )
    

mc_rmsL = [R.get_ave() for R in mc_rms_raveL]
td_rmsL = [R.get_ave() for R in td_rms_raveL]

fig, ax = plt.subplots()

ax.plot(mc_rmsL, 'r-', label='MC')
ax.plot(td_rmsL, 'c-', label='TD(0)')
td_erros = [0.23570226, 0.23500565, 0.14095847, 0.13294523, 0.12816648,
       0.12551351, 0.12472649, 0.12393498, 0.1234716 , 0.12199879,
       0.11964723, 0.11581404, 0.11171416, 0.1092907 , 0.10681323,
       0.10458747, 0.10274803, 0.10058374, 0.09793287, 0.09578209,
       0.09267628, 0.08953371, 0.08651356, 0.08432881, 0.08120382,
       0.07957588, 0.0777723 , 0.07588626, 0.0742679 , 0.07291501,
       0.07222157, 0.07067874, 0.06891229, 0.06738446, 0.06615357,
       0.06443629, 0.06282793, 0.06144512, 0.06040959, 0.05944116,
       0.05866245, 0.05772758, 0.0571357 , 0.05695923, 0.05680691,
       0.05608458, 0.05548595, 0.05491622, 0.05399334, 0.05334945,
       0.05303594, 0.05210484, 0.05104764, 0.05049412, 0.04994111,
       0.04943265, 0.04914664, 0.04871571, 0.04823379, 0.04774282,
       0.04715639, 0.04649857, 0.0458638 , 0.04556575, 0.0451605 ,
       0.04509456, 0.04498466, 0.04484148, 0.04441262, 0.04399574,
       0.04355542, 0.04303882, 0.04261556, 0.04247609, 0.04230481,
       0.0421103 , 0.0419708 , 0.04179976, 0.04166109, 0.04146754,
       0.04110138, 0.04079011, 0.04039504, 0.0399459 , 0.03945678,
       0.03922889, 0.03905436, 0.03873144, 0.03847842, 0.03854734,
       0.0383448 , 0.03821861, 0.03824804, 0.03821737, 0.03814241,
       0.03794948, 0.0376604 , 0.03719871, 0.03700129, 0.0367717 ,
       0.03646475]

mc_erros = [0.32090206, 0.3073737 , 0.28464613, 0.24484697, 0.20757003,
       0.19248474, 0.17944204, 0.16946804, 0.16277973, 0.16181715,
       0.15962668, 0.14989034, 0.14870127, 0.14569844, 0.14070048,
       0.13875578, 0.1351169 , 0.13082869, 0.12741113, 0.12455362,
       0.11893822, 0.11750701, 0.11616459, 0.11504002, 0.11015156,
       0.10608762, 0.10345609, 0.10131899, 0.09932179, 0.09686175,
       0.09588385, 0.09396435, 0.09039547, 0.08835641, 0.08783245,
       0.08641919, 0.08487034, 0.08484189, 0.08392574, 0.0839213 ,
       0.08265648, 0.08223952, 0.08126605, 0.08106522, 0.08053886,
       0.07924668, 0.07798944, 0.07686527, 0.07544847, 0.07444613,
       0.07438756, 0.07416803, 0.07437839, 0.07410547, 0.07356762,
       0.07399262, 0.07407239, 0.07450469, 0.07414103, 0.07357016,
       0.07302586, 0.07206293, 0.07243964, 0.07169077, 0.07059434,
       0.07073674, 0.07040841, 0.06928873, 0.06854812, 0.06866516,
       0.06755258, 0.06750844, 0.06747084, 0.06636694, 0.06624459,
       0.06606405, 0.06570818, 0.06557933, 0.06498521, 0.06439692,
       0.06320915, 0.063198  , 0.06269102, 0.06248714, 0.06260347,
       0.06224086, 0.0621441 , 0.06165326, 0.06171364, 0.06169938,
       0.06165702, 0.06103074, 0.06068596, 0.06105469, 0.06090248,
       0.06063371, 0.06069861, 0.06014864, 0.0596927 , 0.05976133,
       0.05960022]
ax.plot(mc_erros, 'r:', label='np-MC')
ax.plot(td_erros, 'c:', label='np-TD(0)')

ax.legend()
ax.set(title='Figure 6.2 Batch MC & TD(0) Random Walk')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.ylabel('Ave. RMS Error (100 experiments)')
plt.xlabel('Walks / Episodes')
fig.savefig("fig_6_2_mc_td_random_walk_v2.png")
plt.show()
    
