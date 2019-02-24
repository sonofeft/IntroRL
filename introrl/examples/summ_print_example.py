#import introrl
#print(introrl.__file__)

from introrl.environments.env_baseline import EnvBaseline
grid_world = EnvBaseline( mdp_file='Simple_Grid_World' )
grid_world.summ_print()

#car_rental = EnvBaseline( mdp_file='Jacks_Car_Rental_(var_rtn)' )
#car_rental = EnvBaseline( mdp_file='Jacks_Car_Rental_(const_rtn)' )
#car_rental.summ_print()