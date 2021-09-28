from enum import Enum

class Labels(Enum):
    rel_labels = ['msdl_M_state_arg', 'msdl_M_state_mdu', 'msdl_M_state_submdu', 'msdl_M_state_submdu2']
    phys_labels = ['mif_M_n_eng', 'mif_M_n_gb_in', 'mif_M_n_gb_in_1', 'tgd_M_gear_driv']

class Tables(Enum):
    default_default_states = "default_states"
    default_patterns_simple_maneuvers = "patterns_simple_manuvers"
    e_f_sdl_maneuver_messung = "e_f_sdl.maneuver_messung"
    e_f_sdl_maneuver_simulation = "e_f_sdl.maneuver_simulation"

class Limits(Enum):
    UPPER_LIMIT = 200000
    LOWER_LIMIT = 100000
    UPPER_LIMIT_MESSUNG = 150000


# or
# class Tables(Enum):
#     default = {
#         "default_states": "default_states",
#         "patterns_simple_maneuvers": "patterns_simple_manuvers"
#     }
#     e_f_sdl = {
#         "maneuver_messung": "e_f_sdl.maneuver_messung",
#         "maneuver_simulation": "e_f_sdl.maneuver_simulation"
#     }
