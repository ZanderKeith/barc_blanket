# Run every case we are interested in.
# To be looked at tomorrow morning
import os

from openmc.data import atomic_mass

from barc_blanket.utilities import working_directory
from barc_blanket.models.barc_model_final import make_model
from barc_blanket.materials.blanket_depletion import run_coupled_depletion
from barc_blanket.models.materials import flibe, lid, pbli, burner_mixture, pure_ma_llnp, lif_bef, burner_mixture_fixed_mass

# Original 22.63 class
# CASES = {
#     'pure_flibe': {'blanket_material': flibe(),
#                    'name': "Pure FLiBe"},
#     'pure_lid': {'blanket_material': lid(),
#                  "name": "Pure LiD"},
#     'pure_pbli': {'blanket_material': pbli(),
#                   "name": "Pure PbLi"},
#     'waste_01_flibe': {'blanket_material': burner_mixture(0.01, flibe=flibe()),
#                        "name": "FLiBe 1% Full Tank Inventory"},
#     'waste_01_lid': {'blanket_material': burner_mixture(0.01, flibe=lid()),
#                      "name": "LiD 1% Full Tank Inventory"},
#     'waste_01_pbli': {'blanket_material': burner_mixture(0.01, flibe=pbli()),
#                       "name": "PbLi 1% Full Tank Inventory"},
#     'waste_05_flibe': {'blanket_material': burner_mixture(0.05, flibe=flibe()),
#                        "name": "FLiBe 5% Full Tank Inventory"},
#     'waste_05_lid': {'blanket_material': burner_mixture(0.05, flibe=lid()),
#                      "name": "LiD 5% Full Tank Inventory"},
#     'waste_05_pbli': {'blanket_material': burner_mixture(0.05, flibe=pbli()),
#                       "name": "PbLi 5% Full Tank Inventory"},
#     'waste_10_flibe': {'blanket_material': burner_mixture(0.10, flibe=flibe()),
#                        "name": "FLiBe 10% Full Tank Inventory"},
#     'waste_10_lid': {'blanket_material': burner_mixture(0.10, flibe=lid()),
#                      "name": "LiD 10% Full Tank Inventory"},
#     'waste_10_pbli': {'blanket_material': burner_mixture(0.10, flibe=pbli()),
#                       "name": "PbLi 10% Full Tank Inventory"},
# }

# ARPA-E proposal
pure_ma_llnp_05_flibe_burner_mixture = burner_mixture(0.05, tank_contents=pure_ma_llnp(), flibe=lif_bef(0.7, 0.2))
pure_ma_llnp_10_flibe_burner_mixture = burner_mixture(0.10, tank_contents=pure_ma_llnp(), flibe=lif_bef(0.7, 0.2))
pure_ma_llnp_20_flibe_burner_mixture = burner_mixture(0.20, tank_contents=pure_ma_llnp(), flibe=lif_bef(0.7, 0.2))
pure_ma_llnp_05_pbli_burner_mixture = burner_mixture_fixed_mass(pure_ma_llnp(), pbli(), pure_ma_llnp_05_flibe_burner_mixture)
pure_ma_llnp_10_pbli_burner_mixture = burner_mixture_fixed_mass(pure_ma_llnp(), pbli(), pure_ma_llnp_10_flibe_burner_mixture)
pure_ma_llnp_20_pbli_burner_mixture = burner_mixture_fixed_mass(pure_ma_llnp(), pbli(), pure_ma_llnp_20_flibe_burner_mixture)
CASES = {
    "pure_ma_llnp_05_flibe": {"blanket_material": pure_ma_llnp_05_flibe_burner_mixture,
                                "name": "MA LLNP 5% FLiBe"},
    "pure_ma_llnp_10_flibe": {"blanket_material": pure_ma_llnp_10_flibe_burner_mixture,
                                "name": "MA LLNP 10% FLiBe"},
    "pure_ma_llnp_20_flibe": {"blanket_material": pure_ma_llnp_20_flibe_burner_mixture,
                                "name": "MA LLNP 20% FLiBe"},
    "pure_ma_llnp_05_pbli": {"blanket_material": pure_ma_llnp_05_pbli_burner_mixture,
                                "name": "MA LLNP 5% PbLi"},
    "pure_ma_llnp_10_pbli": {"blanket_material": pure_ma_llnp_10_pbli_burner_mixture,
                                "name": "MA LLNP 10% PbLi"},
    "pure_ma_llnp_20_pbli": {"blanket_material": pure_ma_llnp_20_pbli_burner_mixture,
                                "name": "MA LLNP 20% PbLi"},
}

# Debug the absolute mass of the blanket material for each case
for case, config in CASES.items():
    print(f"====================")
    print(f"Case: {case}")
    blanket_material = config['blanket_material']
    atom_densities = blanket_material.get_nuclide_atom_densities()
    atoms = {}
    masses = {}
    for nuclide in atom_densities:
        atoms[nuclide] = atom_densities[nuclide]
        masses[nuclide] = blanket_material.get_mass_density(nuclide)

    total_atoms = sum(atoms.values())
    waste_atoms = sum(atoms[nuclide] for nuclide in pure_ma_llnp().get_nuclides())
    print(f"Waste atom fraction: {waste_atoms / total_atoms}")

    total_mass = sum(masses.values())
    print(f"Total mass density: {total_mass}\t True mass density: {blanket_material.density}")
    waste_mass = sum(masses[nuclide] for nuclide in pure_ma_llnp().get_nuclides())
    print(f"Waste mass density fraction: {waste_mass / total_mass}")
    print(f"====================")

BATCHES = 30
PARTICLES = int(1e3)
PHOTON_TRANSPORT = False

def main():
    for case, config in CASES.items():
        # create a working directory for each case
        os.makedirs(f"depletion_results/{case}", exist_ok=True)
        with working_directory(f"depletion_results/{case}"):
            model_config = {"batches": BATCHES,
                            "particles": PARTICLES,
                            "photon_transport": PHOTON_TRANSPORT,
                            "blanket_material": config['blanket_material']}
            
            model = make_model(model_config)
            model.export_to_model_xml()

            fusion_power = 2.2  # GW
            timesteps_years = [1] * 5 # 1 year timesteps for 5 years

            run_coupled_depletion(model, timesteps_years, fusion_power)

if __name__ == "__main__":
    main()