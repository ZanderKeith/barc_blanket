import os
import openmc
from .create_waste import create_waste_material

# Plasma
def dt_plasma():
    dt_plasma = openmc.Material(name='dt_plasma')
    dt_plasma.add_nuclide('H2', 1.0)
    dt_plasma.add_nuclide('H3', 1.0)
    dt_plasma.set_density('g/cm3', 1e-5)
    return dt_plasma

# FLIBE
def flibe(li6_enrichment=None):
    flibe = openmc.Material(name="flibe")
    flibe.depletable=True
    flibe.add_element("Be", 1.0, "ao")
    flibe.add_element("F", 4.0, "ao")

    if li6_enrichment is None:
        flibe.add_element("Li", 2.0, "ao")
    else:
        flibe.add_element("Li", 2.0, "ao", 
                        enrichment=li6_enrichment, 
                        enrichment_target="Li6", 
                        enrichment_type="ao")

    flibe.set_density("g/cm3", 1.94)
    return flibe

def lif_bef(lif_pct, bef_pct):
    """LiF and BeF2 mixture, based on atom percent. Table 25 of https://www.osti.gov/servlets/purl/5352526
    """

    lif = openmc.Material(name='lif')
    lif.depletable = True
    lif.add_element('Li', 1.0, 'ao')
    lif.add_element('F', 1.0, 'ao')
    lif.set_density('g/cm3', 2.64) # https://en.wikipedia.org/wiki/Lithium_fluoride

    bef = openmc.Material(name='bef')
    bef.depletable = True
    bef.add_element('Be', 1.0, 'ao')
    bef.add_element('F', 2.0, 'ao')
    bef.set_density('g/cm3', 1.99) # https://en.wikipedia.org/wiki/Beryllium_fluoride

    total_pct = lif_pct + bef_pct
    lif_ao = lif_pct / total_pct
    bef_ao = bef_pct / total_pct

    lif_bef = openmc.Material.mix_materials([lif, bef], [lif_ao, bef_ao], percent_type='ao', name='lif_bef')
    lif_bef.depletable = True
    return lif_bef

# Lithium deuteride
def lid():
    lid = openmc.Material(name='lid')
    lid.depletable = True
    lid.add_element('Li', 1.0, 'ao')
    lid.add_nuclide('H2', 1.0, 'ao')
    lid.set_density('g/cm3', 0.82)
    return lid

# Lead lithium
def pbli():
    pbli = openmc.Material(name='pbli')
    pbli.depletable = True
    pbli.add_element('Pb', 84.2, 'ao')
    pbli.add_element('Li', 15.8, 'ao')
    pbli.set_density('g/cm3', 10.2)
    return pbli

# Inconel 718 -
def inconel718():
    inconel718 = openmc.Material(name='inconel718')
    inconel718.depletable = True
    inconel718.add_element('Ni', 53.0, 'wo')
    inconel718.add_element('Cr', 19.06, 'wo')
    inconel718.add_element('Nb', 5.08, 'wo')
    inconel718.add_element('Mo', 3.04, 'wo')
    inconel718.add_element('Ti', 0.93, 'wo')
    inconel718.add_element('Al', 0.52, 'wo')
    inconel718.add_element('Co', 0.11, 'wo')
    inconel718.add_element('Cu', 0.02, 'wo')
    inconel718.add_element('C', 0.021, 'wo')
    inconel718.add_element('Fe', 18.15, 'wo')
    inconel718.set_density('g/cm3', 8.19)
    return inconel718

# Eurofer
def eurofer():
    eurofer = openmc.Material(name='eurofer')
    eurofer.depletable = True
    eurofer.add_element('Cr', 8.99866, 'wo')
    eurofer.add_element('C', 0.109997, 'wo')
    eurofer.add_element('W', 1.5, 'wo')
    eurofer.add_element('V', 0.2, 'wo')
    eurofer.add_element('Ta', 0.07, 'wo')
    eurofer.add_element('B', 0.001, 'wo')
    eurofer.add_element('N', 0.03, 'wo')
    eurofer.add_element('O', 0.01, 'wo')
    eurofer.add_element('S', 0.001, 'wo')
    eurofer.add_element('Fe', 88.661, 'wo')
    eurofer.add_element('Mn', 0.4, 'wo')
    eurofer.add_element('P', 0.005, 'wo')
    eurofer.add_element('Ti', 0.01, 'wo')
    eurofer.set_density('g/cm3', 7.798)
    return eurofer

# V-4Cr-4Ti - pure -(from Segantin TRE https://github.com/SteSeg/tokamak_radiation_environment)
def v4cr4ti():
    v4cr4ti = openmc.Material(name='v4cr4ti')
    v4cr4ti.depletable = True
    v4cr4ti.add_element('V', 0.92, 'wo')
    v4cr4ti.add_element('Cr', 0.04, 'wo')
    v4cr4ti.add_element('Ti', 0.04, 'wo')
    v4cr4ti.set_density('g/cm3', 6.06)
    return v4cr4ti

# Tungsten - pure
def tungsten():
    tungsten = openmc.Material(name='tungsten')
    tungsten.depletable = True
    tungsten.add_element('W', 1.0, 'wo')
    tungsten.set_density('g/cm3', 19.3)
    return tungsten

# Water
def water():
    water = openmc.Material(name='water')
    water.depletable = True
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)
    return water

# Raw tank contents, do however you want to define this
def tank_contents(mixture_name:str):
    """Return the material from the premade tank contents"""

    module_file_path = os.path.dirname(__file__)
    material_xml_path = f"{module_file_path}/../materials/{mixture_name}.xml"

    tank_contents = openmc.Materials.from_xml(material_xml_path)[0]

    return tank_contents

MA_LLNP = ["Np237", "Am241", "Am243", "Cm242", "Cm244", "Tc99", "I129", "Cs135", "Zr93"]

def pure_ma_llnp():
    """PWR spent fuel, minus the uranium
    https://www.cea.fr/english/Documents/scientific-and-economic-publications/nuclear-energy-monographs/CEA_Monograph6_Treatment-recycling-spent-nuclear-fuel_2008_GB.pdf
    """

    kg_per_tonne = {
        "Np237": 0.916,
        "Am241": 0.490,
        "Am243": 0.294,
        "Cm244": 0.011,
        "Se79": 0.008,
        "Zr93": 1.25,
        "Tc99": 1.41,
        "I129": 0.308,
        "Cs135": 0.769,
    }

    total_mass = sum(kg_per_tonne.values())
    mass_fraction = {nuclide: kg_per_tonne[nuclide] / total_mass for nuclide in kg_per_tonne}

    pure_ma_llnp = openmc.Material(name="pure_ma_llnp")
    pure_ma_llnp.depletable = True
    pure_ma_llnp.set_density("g/cm3", 10) # Just a guess
    for nuclide, fraction in mass_fraction.items():
        pure_ma_llnp.add_nuclide(nuclide, fraction, "wo")
    return pure_ma_llnp

# Mixture of tank contents and flibe for the blanket
def burner_mixture(slurry_ratio, percent_type='ao', tank_contents=tank_contents("full_tank_inventory"), flibe=flibe()):
    """Create a mixture of flibe and tank contents for the blanket
    
    Parameters:
    ----------
    slurry_ratio : float
        The 'method' percent of slurry in the blanket
    percent_type : str, optional
        The method to use for the mixture. Default is 'vo', can also use 'wo' and 'ao'.
    tank_contents : openmc.Material, optional
        The tank contents to use in the mixture. Default is natural uranium.
    flibe : openmc.Material, optional
        The FLiBe material to use in the mixture. Default is the standard FLiBe material.
        Can pass in enriched flibe if desired

    Returns:
    -------
    burner_mixture : openmc.Material
        The mixture of FLiBe and tank contents
    
    """
    flibe_ratio = 1 - slurry_ratio

    burner_mixture = openmc.Material.mix_materials(
        [flibe, tank_contents],
        [flibe_ratio, slurry_ratio],
        percent_type=percent_type,
        name="burner_mixture"
    )
    burner_mixture.depletable = True

    return burner_mixture

def burner_mixture_fixed_mass(waste_material, breeder_material, reference_material):
    """Create a mixture where the mass of waste material is the same as in the reference material
    
    Parameters:
    ----------
    waste_material : openmc.Material
        The waste material of fission products and other stuff.
    breeder_material : openmc.Material
        FLiBe, PbLi, etc.
    reference_material : openmc.Material
        The material to match the mass of. Assumed to be a mixture of waste_material and some other breeder_material
    """

    # We are maintaining the mass density of the waste material,
    # and diluting the breeder material according to the volume displaced by the waste material
    
    diluted_waste_mass_density = 0
    for nuclide in waste_material.get_nuclides():
        diluted_waste_mass_density += reference_material.get_mass_density(nuclide)

    tank_volume = 1 # Don't actually know this, just putting it down to make this logic clearer
    waste_mass = diluted_waste_mass_density * tank_volume
    waste_volume = waste_mass / waste_material.density

    breeder_volume = tank_volume - waste_volume

    waste_volume_percent = waste_volume / tank_volume
    breeder_volume_percent = breeder_volume / tank_volume

    # Put the mixture together in terms of volume fractions
    mixture = openmc.Material.mix_materials(
        [waste_material, breeder_material],
        [waste_volume_percent, breeder_volume_percent],
        percent_type='vo',
        name="burner_mixture_fixed_mass"
    )
    mixture.depletable = True
    #mixture.set_density("g/cm3", mixture.density*mass_density_correction_factor)

    return mixture

        
