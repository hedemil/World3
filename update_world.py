import matplotlib.pyplot as plt
from more_itertools import last
import numpy as np

from pyworld3 import World3, world3
from pyworld3.utils import plot_world_variables

params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

def run_world3_simulation(year_min, year_max, dt=1, initial_conditions=None, prev_run_data=None, first=True, k_index=1):
    world3 = World3(year_max=year_max, year_min=year_min, dt=dt, prev_run_data=prev_run_data)
    world3.set_world3_control()
    world3.init_world3_constants() if initial_conditions is None else world3.init_world3_constants(**initial_conditions)    
    world3.init_world3_variables() if prev_run_data is None else world3.init_world3_variables(prev_run_data=prev_run_data)
    world3.set_world3_table_functions()
    world3.set_world3_delay_functions() if prev_run_data is None else world3.set_world3_delay_functions(prev_run_data=prev_run_data) 
    if first:
        k = world3.run_world3(fast=False)
        return world3, k
    else:
        world3.run_world3(fast=False, first=False, k_index=k_index)
        return world3

def extract_values(world3):
    variables = ["al", "pal", "uil", "lfert", "ai", "pfr", "ic", "sc", "cuf", "ppol", "p1", "p2", "p3", "p4", "frsn", "nr"]
    last_values = {var: getattr(world3, var)[-1] for var in variables}
    return last_values

def prepare_initial_conditions(last_values):
    # Map the extracted values to the expected initial condition keys
    initial_conditions = {
        "ali": last_values["al"],
        "pali": last_values["pal"],
        "uili": last_values["uil"],
        "lferti": last_values["lfert"],
        "ici": last_values["ic"],
        "sci": last_values["sc"],
        "ppoli": last_values["ppol"],
        "p1i": last_values["p1"],
        "p2i": last_values["p2"],
        "p3i": last_values["p3"],
        "p4i": last_values["p4"],
        "nri": last_values["nr"]
        # Note: You might need to adjust the keys based on the expected parameter names in init_world3_constants
    }
    return initial_conditions

def main():
    # Run the first simulation
    world3_first, k_index = run_world3_simulation(year_min=1900, year_max=2000)
    
    # Extract last values and full arrays from the first simulation
    last_values = extract_values(world3_first)
    initial_conditions = prepare_initial_conditions(last_values)

    prev_run_data = {
        'population': {
            'p1': world3_first.p1,
            'p2': world3_first.p2,
            'p3': world3_first.p3,
            'p4': world3_first.p4,
            'pop': world3_first.pop,
            'mat1': world3_first.mat1,
            'mat2': world3_first.mat2,
            'mat3': world3_first.mat3,
            'd': world3_first.d,
            'd1': world3_first.d1,
            'd2': world3_first.d2,
            'd3': world3_first.d3,
            'd4': world3_first.d4,
            'cdr': world3_first.cdr,
            'ehspc': world3_first.ehspc,
            'fpu': world3_first.fpu,
            'hsapc': world3_first.hsapc,
            'le': world3_first.le,
            'lmc': world3_first.lmc,
            'lmf': world3_first.lmf,
            'lmhs': world3_first.lmhs,
            'lmhs1': world3_first.lmhs1,
            'lmhs2': world3_first.lmhs2,
            'lmp': world3_first.lmp,
            'm1': world3_first.m1,
            'm2': world3_first.m2,
            'm3': world3_first.m3,
            'm4': world3_first.m4,
            'b': world3_first.b,
            'aiopc': world3_first.aiopc,
            'cbr': world3_first.cbr,
            'cmi': world3_first.cmi,
            'cmple': world3_first.cmple,
            'diopc': world3_first.diopc,
            'dtf': world3_first.dtf,
            'dcfs': world3_first.dcfs,
            'fcapc': world3_first.fcapc,
            'fce': world3_first.fce,
            'fcfpc': world3_first.fcfpc,
            'fie': world3_first.fie,
            'fm': world3_first.fm,
            'frsn': world3_first.frsn,
            'fsafc': world3_first.fsafc,
            'mtf': world3_first.mtf,
            'nfc': world3_first.nfc,
            'ple': world3_first.ple,
            'sfsn': world3_first.sfsn,
            'tf': world3_first.tf,
        },
        'population_delay': {
            'smooth_hsapc': world3_first.smooth_hsapc.out_arr,
            'smooth_iopc': world3_first.smooth_iopc.out_arr,
            'dlinf3_le': world3_first.dlinf3_le.out_arr,
            'dlinf3_iopc': world3_first.dlinf3_iopc.out_arr,
            'dlinf3_fcapc': world3_first.dlinf3_fcapc.out_arr,
        },
        'capital': {
            'ic': world3_first.ic,
            'io': world3_first.io,
            'icdr': world3_first.icdr,
            'icir': world3_first.icir,
            'icor': world3_first.icor,
            'iopc': world3_first.iopc,
            'alic': world3_first.alic,
            'fioac': world3_first.fioac,
            'fioacv': world3_first.fioacv,
            'fioai': world3_first.fioai,
            'sc': world3_first.sc,
            'so': world3_first.so,
            'scdr': world3_first.scdr,
            'scir': world3_first.scir,
            'scor': world3_first.scor,
            'sopc': world3_first.sopc,
            'alsc': world3_first.alsc,
            'isopc': world3_first.isopc,
            'fioas': world3_first.fioas,
            'j': world3_first.j,
            'jph': world3_first.jph,
            'jpicu': world3_first.jpicu,
            'jpscu': world3_first.jpscu,
            'lf': world3_first.lf,
            'cuf': world3_first.cuf,
            'luf': world3_first.luf,
            'lufd': world3_first.lufd,
            'pjas': world3_first.pjas,
            'pjis': world3_first.pjis,
            'pjss': world3_first.pjss,
            'smooth_luf': world3_first.smooth_luf.out_arr,
        },
        'agriculture': {
            'al': world3_first.al,
            'pal': world3_first.pal,
            'dcph': world3_first.dcph,
            'f': world3_first.f,
            'fpc': world3_first.fpc,
            'fioaa': world3_first.fioaa,
            'ifpc': world3_first.ifpc,
            'ldr': world3_first.ldr,
            'lfc': world3_first.lfc,
            'tai': world3_first.tai,
            'ai': world3_first.ai,
            'aiph': world3_first.aiph,
            'alai': world3_first.alai,
            'cai': world3_first.cai,
            'ly': world3_first.ly,
            'lyf': world3_first.lyf,
            'lymap': world3_first.lymap,
            'lymc': world3_first.lymc,
            'fiald': world3_first.fiald,
            'mlymc': world3_first.mlymc,
            'mpai': world3_first.mpai,
            'mpld': world3_first.mpld,
            'uil': world3_first.uil,
            'all': world3_first.all,
            'llmy': world3_first.llmy,
            'ler': world3_first.ler,
            'lrui': world3_first.lrui,
            'uilpc': world3_first.uilpc,
            'uilr': world3_first.uilr,
            'lfert': world3_first.lfert,
            'lfd': world3_first.lfd,
            'lfdr': world3_first.lfdr,
            'lfr': world3_first.lfr,
            'lfrt': world3_first.lfrt,
            'falm': world3_first.falm,
            'fr': world3_first.fr,
            'pfr': world3_first.pfr,
            'smooth_cai': world3_first.smooth_cai.out_arr,
            'in_cai': world3_first.smooth_cai.in_arr,
            'smooth_fr': world3_first.smooth_fr.out_arr,
            'in_fr': world3_first.smooth_fr.in_arr,
            },
        'pollution': {
            'ppol': world3_first.ppol,
            'ppolx': world3_first.ppolx,
            'ppgao': world3_first.ppgao,
            'ppgio': world3_first.ppgio,
            'ppgf': world3_first.ppgf,
            'ppgr': world3_first.ppgr,
            'ppapr': world3_first.ppapr,
            'ppasr': world3_first.ppasr,
            'pptd': world3_first.pptd,
            'ahl': world3_first.ahl,
            'ahlm': world3_first.ahlm,
            'delay3_ppgr': world3_first.delay3_ppgr.out_arr,
            },
        'resource': {
            'nr': world3_first.nr,
            'nrfr': world3_first.nrfr,
            'nruf': world3_first.nruf,
            'nrur': world3_first.nrur,
            'pcrum': world3_first.pcrum,
            'fcaor': world3_first.fcaor,
            },
        'time': world3_first.time,
        'n': world3_first.n,
        }
    
    # Run the second simulation with initial conditions derived from the first simulation
    world3_second = run_world3_simulation(year_min=2000, year_max=2100, prev_run_data=prev_run_data, first=False, k_index=k_index)
    

    # Plot the combined results
    plot_world_variables(
        world3_second.time,
        [world3_second.fpc, world3_second.fr, world3_second.pop, world3_second.ppolx],
        ["FPC", "FR", "POP", "PPOLX"],
        [[0, 2e3], [0, 5], [0, 10e9], [0, 20]],
        figsize=(10, 7),
        title="World3 Simulation from 1900 to 2200, paused at 2000"
    )
    plt.show()

if __name__ == "__main__":
    main()

