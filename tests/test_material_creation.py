import numpy as np

from barc_blanket.utilities import working_directory
from barc_blanket.models.barc_model_final import make_model
from barc_blanket.materials.blanket_depletion import run_coupled_depletion
from barc_blanket.models.materials import flibe, lid, pbli, burner_mixture, pure_ma_llnp, lif_bef, burner_mixture_fixed_mass


class TestBurnerMixtureFixedMass:

    def test_pure_ma_llnp_05_flibe_to_pbli(self):
        pure_ma_llnp_05_flibe_burner_mixture = burner_mixture(0.05, tank_contents=pure_ma_llnp(), flibe=lif_bef(0.7, 0.2))
        pure_ma_llnp_05_pbli_burner_mixture = burner_mixture_fixed_mass(pure_ma_llnp(), pbli(), pure_ma_llnp_05_flibe_burner_mixture)

        # Compare the mass density of waste in each mixture
        # If they are the same, that means we will have the same mass of waste when filling the blanket volume

        original_mass_density = 0
        for nuclide in pure_ma_llnp().get_nuclides():
            original_mass_density += pure_ma_llnp_05_flibe_burner_mixture.get_mass_density(nuclide)
        
        new_mass_density = 0
        for nuclide in pure_ma_llnp().get_nuclides():
            new_mass_density += pure_ma_llnp_05_pbli_burner_mixture.get_mass_density(nuclide)

        assert np.isclose(original_mass_density, new_mass_density)

    def test_pure_ma_llnp_20_flibe_to_pbli(self):
        pure_ma_llnp_20_flibe_burner_mixture = burner_mixture(0.20, tank_contents=pure_ma_llnp(), flibe=lif_bef(0.7, 0.2))
        pure_ma_llnp_20_pbli_burner_mixture = burner_mixture_fixed_mass(pure_ma_llnp(), pbli(), pure_ma_llnp_20_flibe_burner_mixture)

        # Compare the mass density of waste in each mixture
        # If they are the same, that means we will have the same mass of waste when filling the blanket volume

        original_mass_density = 0
        for nuclide in pure_ma_llnp().get_nuclides():
            original_mass_density += pure_ma_llnp_20_flibe_burner_mixture.get_mass_density(nuclide)
        
        new_mass_density = 0
        for nuclide in pure_ma_llnp().get_nuclides():
            new_mass_density += pure_ma_llnp_20_pbli_burner_mixture.get_mass_density(nuclide)

        # Check we have the same mass density of waste in the new mixture
        assert np.isclose(original_mass_density, new_mass_density)

        # Make sure the mixture changes in mass density as expected
        if pure_ma_llnp().density > pbli().density:
            assert pure_ma_llnp_20_pbli_burner_mixture.density > pbli().density
        else:
            assert pure_ma_llnp_20_pbli_burner_mixture.density < pbli().density