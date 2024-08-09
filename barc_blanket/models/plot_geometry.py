import matplotlib.pyplot as plt

def plot_geometry(model):
    fig, ax = plt.subplots(1,2, figsize=[12,6])

    universe = model.geometry.root_universe

    universe.plot(pixels=5000000,
                  width=(800,450),
                  origin=(480, 150, 0),
                  color_by='cell',
                axes=ax[0]
                )

    universe.plot(width=(800.0, 800.0), 
                origin=(480.0, 0.0, 0.1), 
                basis='xz',
                pixels=5000000,
                color_by='cell',
                axes=ax[1]
                )

    ax[0].set_title('Top View')
    ax[0].set_xlabel('x(cm)')
    ax[0].set_ylabel('y(cm)')
    ax[1].set_title('Azimuthal View')
    ax[1].set_xlabel('x(cm)')
    ax[1].set_ylabel('z(cm)')
    ax[1].legend(loc='upper right')
    fig.tight_layout()

    return fig

def presentation_plots(model):
    fig, ax = plt.subplots(1,2, figsize=[8,6])

    universe = model.geometry.root_universe

    colors={
        1: 'thistle', # Plasma
        2: 'slategrey', # First wall
        3: 'palegreen', # FLiBe
        4: 'slategrey', # First wall
        5: 'darkorange', # Vacuum Vessel
        6: 'palegreen', # FLiBe
        7: 'darkorange', # Blanket vessel
        8: 'grey', # Neutron shield
        9: 'steelblue', # Magnet case
        10: 'goldenrod', # TF coil
        11: 'coral', # TF coil
        12: 'steelblue', # Magnet case
        13: 'red',
        14: 'red',
        15: 'red',
        16: 'red',
    }

    universe.plot(width=(400.0, 550.0), 
                origin=(680, 0, 300), 
                basis='xz',
                pixels=5000000,
                color_by='cell',
                axes=ax[1],
                colors=colors
                )

    ax[1].set_title('Radial View of OpenMC Model')
    ax[1].set_xlabel('Minor Radius [cm]', fontsize=20)
    ax[1].set_ylabel('z [cm]', fontsize=20)
    #ax.legend(loc='upper right')
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    # Set y tick font size
    ax[1].tick_params(axis='y', labelsize=18)
    # Subtract 480 from all x ticks
    xticks = ax[1].get_xticks()
    xticks = [int(x - 480) for x in xticks]
    ax[1].set_xticklabels(xticks, fontsize=18)


    universe.plot(width=(10.0, 10.0), 
                origin=(480, 0, 236.9), 
                basis='xz',
                pixels=5000000,
                color_by='cell',
                axes=ax[0],
                colors=colors
                )
    
    # put y ticks on the right
    ax[0].yaxis.set_label_position("right")
    ax[0].yaxis.tick_right()
    ax[0].set_ylabel('z [cm]', fontsize=24)
    ax[0].tick_params(axis='y', labelsize=22)
    # Remove x ticks
    ax[0].set_xticks([])

    # Make figure tight layout
    fig.tight_layout()

    return fig