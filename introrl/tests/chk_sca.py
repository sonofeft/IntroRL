from introrl.policy import Policy
from introrl.black_box_sims.racetrack_1_sim import RaceTrack_1

RT = RaceTrack_1()
sca = Policy(environment=RT)

sca.add_state_action( (25, 7, 0, 1) )

sca.set_action_prob( (25, 7, 0, 1), (1,1), prob=1.0)
            
#sca.summ_print()

SA = sca.get_SA_object((25, 7, 0, 1))
print( SA )
SA.summ_print()
print('-'*55)

sca.set_policy_from_piD( RT.get_default_policy_desc_dict() )
SA = sca.get_SA_object((25, 7, 0, 1))
print( SA )
SA.summ_print()
