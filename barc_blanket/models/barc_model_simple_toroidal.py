import os
import openmc
import numpy as np

from .materials import dt_plasma, flibe, burner_mixture, v4cr4ti, tungsten

# Default model parameters
# TODO: this all assumes a circular cross-section, which is not necessarily the case
# Must determine if this is a reasonable assumption
# See 'simple_toroidal.png' for a diagram of the geometry

DEFAULT_PARAMETERS = {
    'major_radius': 450,            # All dimensions are in cm
    'plasma_minor_radius': 100,
    'sol_width': 2,
    'first_wall_thickness': 0.1,          # How thick the plasma facing material is
    'vacuum_vessel_thickness': 1,         # How thick the vacuum vessel is
    'cooling_channel_width': 1,           # Width of the flowing coolant
    'cooling_vessel_thickness': 2,        # How thick the cooling vessel is
    'blanket_width': 130,                 # Width of the bulk molten salt blanket
    'blanket_vessel_thickness': 8,        # How thick the blanket vessel is

    'section_angle': 45,            # Angle of the toroidal section in degrees

    'li6_enrichment': 0.076,        # atom% enrichment of Li6 in the FLiBe
    'slurry_ratio': 0.01            # atom% slurry in the burner blanket
}

def make_model(new_model_config=None):
    """Create an OpenMC model using the given configuration
    
    Parameters:
    ----------
    new_model_config : dict, optional
        Dictionary containing the model configuration.
        If not provided, the values listed in DEFAULT_PARAMETERS will be used.

    Returns:
    -------
    model : openmc.Model
        An OpenMC model object
    """

    if new_model_config is None:
        model_config = DEFAULT_PARAMETERS
    else:
        model_config = new_model_config.copy()
        for key in DEFAULT_PARAMETERS:
            if key not in new_model_config:
                model_config[key] = DEFAULT_PARAMETERS[key]

    #####################
    ## Assign Materials##
    #####################

    plasma_material = dt_plasma()
    first_wall_material = tungsten()
    vacuum_vessel_material = v4cr4ti()
    flibe_material = flibe(model_config['li6_enrichment'])
    cooling_channel_material = burner_mixture(model_config['slurry_ratio'], flibe=flibe_material)
    cooling_vessel_material = v4cr4ti()
    blanket_material = burner_mixture(model_config['slurry_ratio'], flibe=flibe_material)
    blanket_vessel_material = v4cr4ti()
    
    #####################
    ## Define Geometry ##
    #####################

    R = model_config['major_radius']
    a = model_config['plasma_minor_radius']
    sol_width = model_config['sol_width']
    first_wall_thickness = model_config['first_wall_thickness']

    vacuum_vessel_thickness = model_config['vacuum_vessel_thickness']
    cooling_channel_width = model_config['cooling_channel_width']
    cooling_vessel_thickness = model_config['cooling_vessel_thickness']
    blanket_width = model_config['blanket_width']
    blanket_vessel_thickness = model_config['blanket_vessel_thickness']

    plasma_surface = openmc.ZTorus(x0=0,y0=0,z0=0,a=R,b=a,c=a)
    
    first_wall_inner_radius = a + sol_width
    first_wall_inner_surface = openmc.ZTorus(x0=0,y0=0,z0=0,a=R,b=first_wall_inner_radius,c=first_wall_inner_radius)

    vacuum_vessel_inner_radius = first_wall_inner_radius + first_wall_thickness
    vacuum_vessel_outer_radius = vacuum_vessel_inner_radius + vacuum_vessel_thickness
    vacuum_vessel_inner_surface = openmc.ZTorus(x0=0,y0=0,z0=0,a=R,b=vacuum_vessel_inner_radius,c=vacuum_vessel_inner_radius)
    vacuum_vessel_outer_surface = openmc.ZTorus(x0=0,y0=0,z0=0,a=R,b=vacuum_vessel_outer_radius,c=vacuum_vessel_outer_radius)

    cooling_vessel_inner_radius = vacuum_vessel_outer_radius+cooling_channel_width
    cooling_vessel_outer_radius = cooling_vessel_inner_radius + cooling_vessel_thickness
    cooling_vessel_inner_surface = openmc.ZTorus(x0=0,y0=0,z0=0,a=R,b=cooling_vessel_inner_radius,c=cooling_vessel_inner_radius)
    cooling_vessel_outer_surface = openmc.ZTorus(x0=0,y0=0,z0=0,a=R,b=cooling_vessel_outer_radius,c=cooling_vessel_outer_radius)

    blanket_vessel_inner_radius = cooling_vessel_outer_radius+blanket_width
    blanket_vessel_outer_radius = blanket_vessel_inner_radius + blanket_vessel_thickness
    blanket_vessel_inner_surface = openmc.ZTorus(x0=0,y0=0,z0=0,a=R,b=blanket_vessel_inner_radius,c=blanket_vessel_inner_radius)
    blanket_vessel_outer_surface = openmc.ZTorus(x0=0,y0=0,z0=0,a=R,b=blanket_vessel_outer_radius,c=blanket_vessel_outer_radius)

    bounding_sphere_surface = openmc.Sphere(r=2*R, boundary_type="vacuum")

    # Make two planes to cut the torus into a section
    # Angle follows right hand rule around z axis (https://www.desmos.com/3d/214a6bb908)
    section_angle_rad = np.radians(model_config['section_angle'])
    x_coeff, y_coeff = np.sin(section_angle_rad), -np.cos(section_angle_rad)
    xz_plane = openmc.Plane(a=0, b=1, boundary_type='periodic')
    angled_plane = openmc.Plane(a=x_coeff, b=y_coeff, boundary_type='periodic')
    xz_plane.periodic_surface = angled_plane
    torus_section = +xz_plane & -angled_plane

    plasma_cell = openmc.Cell(
        name='plasma_cell',
        region=-plasma_surface & torus_section,
        fill=plasma_material
    )

    sol_cell = openmc.Cell(
        name='sol_cell',
        region=+plasma_surface & -first_wall_inner_surface & torus_section,
        fill=None
    )

    first_wall_cell = openmc.Cell(
        name='first_wall_cell',
        region=+first_wall_inner_surface & -vacuum_vessel_inner_surface & torus_section,
        fill=first_wall_material
    )

    vacuum_vessel_cell = openmc.Cell(
        name='vacuum_vessel_cell',
        region=+vacuum_vessel_inner_surface & -vacuum_vessel_outer_surface & torus_section,
        fill=vacuum_vessel_material
    )

    cooling_channel_cell = openmc.Cell(
        name='cooling_channel_cell',
        region=+vacuum_vessel_outer_surface & -cooling_vessel_inner_surface & torus_section,
        fill=cooling_channel_material
    )

    cooling_vessel_cell = openmc.Cell(
        name='cooling_vessel_cell',
        region=+cooling_vessel_inner_surface & -cooling_vessel_outer_surface & torus_section,
        fill=cooling_vessel_material
    )

    blanket_cell = openmc.Cell(
        name='blanket_cell',
        region=+cooling_vessel_outer_surface & -blanket_vessel_inner_surface & torus_section,
        fill=blanket_material
    )

    blanket_vessel_cell = openmc.Cell(
        name='blanket_vessel_cell',
        region=+blanket_vessel_inner_surface & -blanket_vessel_outer_surface & torus_section,
        fill=blanket_vessel_material
    )

    bounding_sphere_cell = openmc.Cell(
        name='bounding_sphere_cell',
        region=+blanket_vessel_outer_surface & -bounding_sphere_surface & torus_section,
        fill=None
    )

    universe = openmc.Universe()
    universe.add_cell(plasma_cell)
    universe.add_cell(sol_cell)
    universe.add_cell(first_wall_cell)
    universe.add_cell(vacuum_vessel_cell)
    universe.add_cell(cooling_channel_cell)
    universe.add_cell(cooling_vessel_cell)
    universe.add_cell(blanket_cell)
    universe.add_cell(blanket_vessel_cell)
    universe.add_cell(bounding_sphere_cell)
    geometry = openmc.Geometry(universe)

    #####################
    ## Define Settings ##
    #####################

    source = openmc.IndependentSource()
    source.particle = 'neutron'
    radius = openmc.stats.Discrete([R], [1]) # centered at major radius
    z_values = openmc.stats.Discrete([0], [1])
    angle = openmc.stats.Uniform(a=np.radians(0), b=np.radians(360))
    source.space = openmc.stats.CylindricalIndependent(
        r=radius, phi=angle, z=z_values, origin=(0., 0., 0.))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.muir(e0=14.08e6, m_rat=5, kt=20000)

    settings = openmc.Settings(run_mode='fixed source')
    settings.photon_transport = False
    settings.source = source
    settings.batches = 50
    settings.particles = int(1e5) # modify this to shorten simulation, default was 1e6 
    settings.statepoint = {'batches': [
        5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    settings.output = {'tallies': True}

    #####################
    ## Define Tallies  ##
    #####################

    blanket_cell_filter = openmc.CellFilter([blanket_cell])
    energy_filter = openmc.EnergyFilter(np.logspace(0,7)) # 1eV to 100MeV

    # mesh tally - flux
    tally1 = openmc.Tally(tally_id=1, name="flux_blanket")
    tally1.filters = [blanket_cell_filter,energy_filter]
    tally1.scores = ["flux"]

    # tbr
    tally2 = openmc.Tally(tally_id=2, name="tbr")
    tally2.filters = [blanket_cell_filter]
    tally2.scores = ["(n,Xt)"]

    #power deposition - heating-local
    tally3 = openmc.Tally(tally_id=3, name="heating_burner")
    tally3.filters = [blanket_cell_filter]
    tally3.scores = ["heating-local"]

    tallies = openmc.Tallies([tally1,tally2,tally3]) 

    # Create model
    model = openmc.Model(geometry=geometry,
                         settings=settings, 
                         tallies=tallies)

    return model
