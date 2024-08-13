import openmc.deplete
from run_all_cases import CASES

import openmc
import matplotlib.pyplot as plt

from barc_blanket.utilities import working_directory
from barc_blanket.materials.blanket_depletion import postprocess_coupled_depletion, plot_results
from barc_blanket.models.barc_model_final import BLANKET_MATERIAL_ID

plt.figure(figsize=(8, 6))
plt.xlim([0, 5])
plt.ylim([0, 1.2])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Time [years]", fontsize=16)
plt.ylabel("TBR", fontsize=16)
plt.title("TBR Over Time", fontsize=18)
plt.axhline(y=1.0, color='k', linestyle='--', label='Breakeven')

for case, config in CASES.items():
    try:
        with working_directory(f"depletion_results/{case}"):
            results = openmc.deplete.Results("depletion_results.h5")
            time_days = results.get_times()
            time_years = time_days / 365

            tbr_at_time = []
            for i, time in enumerate(time_years):
                try:
                    step_results = openmc.StatePoint(f"openmc_simulation_n{i}.h5")
                    step_tbr = step_results.tallies[2].mean[0][0][0] + step_results.tallies[2].mean[1][0][0]
                    tbr_at_time.append(step_tbr)
                except Exception as e:
                    print(f"Error in case {case} at time {time}: {e}")
                    tbr_at_time.append(None)
            
            plt.plot(time_years, tbr_at_time, label=config['name'], linewidth=2)

    except Exception as e:
        print(f"Error in case {case}: {e}")
        continue

plt.legend()
plt.savefig(f"tbr_over_time.png")
