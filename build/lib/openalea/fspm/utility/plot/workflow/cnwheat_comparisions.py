import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

#delta_t_simuls = 1509
delta_t_simuls = 0


def phloem(pdf, df_current_organs, df_ref_organs, meteo_data, tmin, tmax):
    fig, axs = plt.subplots(2, 2)

    # phloem sucrose & AA

    # 1
    axs[0, 0].plot(df_current_organs[(df_current_organs.organ == 'phloem')]['t'], df_current_organs[(df_current_organs.organ == 'phloem')]['Conc_Sucrose'], label='current')
    axs[0, 0].plot(df_ref_organs[(df_ref_organs.organ == 'phloem')]['t'], df_ref_organs[(df_ref_organs.organ == 'phloem')]['Conc_Sucrose'], label='Marion')
    axs[0, 0].legend()
    axs[0, 0].set_xlim(tmin, tmax)
    axs[0, 0].set_ylabel('Concentration sucrose (µmol g-1)')

    # 2
    axs[0, 1].plot(df_current_organs[(df_current_organs.organ == 'phloem')]['t'], df_current_organs[(df_current_organs.organ == 'phloem')]['sucrose'], label='current')
    axs[0, 1].plot(df_ref_organs[(df_ref_organs.organ == 'phloem')]['t'], df_ref_organs[(df_ref_organs.organ == 'phloem')]['sucrose'], label='Marion')
    axs[0, 1].legend()
    axs[0, 1].set_xlim(tmin, tmax)
    # axs[0, 1].set_ylim(0, 500)
    axs[0, 1].set_ylabel('Amount of sucrose (µmol C)')

    # 3
    axs[1, 0].plot(df_current_organs[(df_current_organs.organ == 'phloem')]['t'], df_current_organs[(df_current_organs.organ == 'phloem')]['Conc_Amino_Acids'], label='current')
    axs[1, 0].plot(df_ref_organs[(df_ref_organs.organ == 'phloem')]['t'], df_ref_organs[(df_ref_organs.organ == 'phloem')]['Conc_Amino_Acids'], label='Marion')
    axs[1, 0].legend()
    axs[1, 0].set_xlim(tmin, tmax)
    # axs[1, 0].set_ylim(0, 200)
    axs[1, 0].set_ylabel('Concentration amino acids (µmol g-1)')

    ax2 = axs[1, 0].twiny()
    ax2.set_xticks(axs[1, 0].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 0].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))

    # 4
    axs[1, 1].plot(df_current_organs[(df_current_organs.organ == 'phloem')]['t'], df_current_organs[(df_current_organs.organ == 'phloem')]['amino_acids'], label='current')
    axs[1, 1].plot(df_ref_organs[(df_ref_organs.organ == 'phloem')]['t'], df_ref_organs[(df_ref_organs.organ == 'phloem')]['amino_acids'], label='Marion')
    axs[1, 1].legend()
    axs[1, 1].set_xlim(tmin, tmax)
    # axs[1, 1].set_ylim(0, 25)
    axs[1, 1].set_ylabel('amino acids (µmol N)')

    ax2 = axs[1, 1].twiny()
    ax2.set_xticks(axs[1, 1].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 1].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))

    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

def photosynthesis(pdf, df_current_axes, df_ref_axes):
    df_current_axes['day'] = df_current_axes['t'] // 24 +1
    df_ref_axes['day'] = df_ref_axes['t'] // 24 +1

    fig, axis = plt.subplots()
    axis.plot(df_current_axes['day'].unique(), df_current_axes.groupby('day')['Total_Photosynthesis'].sum(), label='current')
    axis.plot(df_ref_axes['day'].unique(), df_ref_axes.groupby('day')['Total_Photosynthesis'].sum(), label='Marion')

    axis.set_xlabel('Time (day)')
    axis.set_ylabel('Total Photosynthesis µmol C')
    axis.legend()

    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

def roots(pdf, df_current_organs, df_ref_organs, meteo_data, tmin, tmax):
    fig, axs = plt.subplots(2, 2)

    # phloem sucrose & AA

    # 1
    axs[0, 0].plot(df_current_organs[(df_current_organs.organ == 'roots')]['t'], df_current_organs[(df_current_organs.organ == 'roots')]['Conc_Sucrose'], label='current')
    axs[0, 0].plot(df_ref_organs[(df_ref_organs.organ == 'roots')]['t'], df_ref_organs[(df_ref_organs.organ == 'roots')]['Conc_Sucrose'], label='Marion')
    axs[0, 0].legend()
    axs[0, 0].set_xlim(tmin, tmax)
    axs[0, 0].set_ylabel('Concentration sucrose (µmol g-1)')

    # 2
    axs[0, 1].plot(df_current_organs[(df_current_organs.organ == 'roots')]['t'], df_current_organs[(df_current_organs.organ == 'roots')]['sucrose'], label='current')
    axs[0, 1].plot(df_ref_organs[(df_ref_organs.organ == 'roots')]['t'], df_ref_organs[(df_ref_organs.organ == 'roots')]['sucrose'], label='Marion')
    axs[0, 1].legend()
    axs[0, 1].set_xlim(tmin, tmax)
    axs[0, 1].set_ylim(0, 1000)
    axs[0, 1].set_ylabel('Amount of sucrose (µmol C)')

    # 3
    axs[1, 0].plot(df_current_organs[(df_current_organs.organ == 'roots')]['t'], df_current_organs[(df_current_organs.organ == 'roots')]['Conc_Nitrates'], label='current')
    axs[1, 0].plot(df_ref_organs[(df_ref_organs.organ == 'roots')]['t'], df_ref_organs[(df_ref_organs.organ == 'roots')]['Conc_Nitrates'], label='Marion')
    axs[1, 0].legend()
    axs[1, 0].set_xlim(tmin, tmax)
    # axs[1, 0].set_ylim(0, 200)
    axs[1, 0].set_ylabel('Concentration nitrates (µmol g-1)')

    ax2 = axs[1, 0].twiny()
    ax2.set_xticks(axs[1, 0].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 0].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))

    # 4
    axs[1, 1].plot(df_current_organs[(df_current_organs.organ == 'roots')]['t'], df_current_organs[(df_current_organs.organ == 'roots')]['Conc_cytokinins'], label='current')
    axs[1, 1].plot(df_ref_organs[(df_ref_organs.organ == 'roots')]['t'], df_ref_organs[(df_ref_organs.organ == 'roots')]['Conc_cytokinins'], label='Marion')
    axs[1, 1].legend()
    axs[1, 1].set_xlim(tmin, tmax)
    axs[1, 1].set_ylabel('Conc_cytokinins (µmol N)')

    ax2 = axs[1, 1].twiny()
    ax2.set_xticks(axs[1, 1].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 1].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))

    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

def dry_mass(pdf, df_current_axes, df_ref_axes, df_current_organs, df_ref_organs, meteo_data, tmin, tmax):
    fig, axs = plt.subplots(2, 2, sharex=True)

    # Dry mass shoot
    axs[0, 0].plot(df_current_axes['t'], df_current_axes['sum_dry_mass_shoot'], label='current')
    axs[0, 0].plot(df_ref_axes['t'], df_ref_axes['sum_dry_mass_shoot'], label='Marion')
    axs[0, 0].legend()
    axs[0, 0].set_xlim(tmin, tmax)
    # axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_ylabel('Dry mass shoot (g)')

    # Dry mass roots
    axs[0, 1].plot(df_current_axes['t'], df_current_axes['sum_dry_mass_roots'], label='current')
    axs[0, 1].plot(df_ref_axes['t'], df_ref_axes['sum_dry_mass_roots'], label='Marion')
    axs[0, 1].legend()
    axs[0, 1].set_xlim(tmin, tmax)
    # axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_ylabel('Dry mass roots (g)')

    # mstruct shoot
    axs[1, 0].plot(df_current_axes['t'], df_current_axes['mstruct_shoot'], label='current')
    axs[1, 0].plot(df_ref_axes['t'], df_ref_axes['mstruct_shoot'], label='Marion')
    axs[1, 0].legend()
    axs[1, 0].set_xlim(tmin, tmax)
    # axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_ylabel('mstruct shoot (g)')
    ax2 = axs[1, 0].twiny()
    ax2.set_xticks(axs[1, 0].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 0].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))

    # mstruct roots
    axs[1, 1].plot(df_current_organs[df_current_organs['organ'] == 'roots']['t'], df_current_organs[df_current_organs['organ'] == 'roots']['mstruct'], label='current')
    axs[1, 1].plot(df_ref_organs[df_ref_organs['organ'] == 'roots']['t'], df_ref_organs[df_ref_organs['organ'] == 'roots']['mstruct'], label='Marion')
    axs[1, 1].legend()
    axs[1, 1].set_xlim(tmin, tmax)
    # axs[1, 1].set_ylim(0, 0.75)
    axs[1, 1].set_ylabel('mstruct roots (g)')
    ax2 = axs[1, 1].twiny()
    ax2.set_xticks(axs[1, 1].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 1].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))

    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # shoot : root
    fig, axis = plt.subplots()
    axis.plot(df_current_axes['t'], df_current_axes['shoot_roots_ratio'], label='current')
    axis.plot(df_ref_axes['t'], df_ref_axes['shoot_roots_ratio'], label='Marion')
    axis.legend()
    axis.set_xlim(tmin, tmax)
    axis.set_ylim(0, 2)
    axis.set_ylabel('shoot : root ratio')

    ax2 = axis.twiny()
    ax2.set_xticks(axis.get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axis.get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))
    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()


def N_mass(pdf, df_current_axes, df_ref_axes, df_current_organs, df_ref_organs, meteo_data, tmin, tmax):
    fig, axs = plt.subplots(2, 2, sharex=True)

    # % N axis
    axs[0, 0].plot(df_current_axes['t'], df_current_axes['N_content'], label='current')
    axs[0, 0].plot(df_ref_axes['t'], df_ref_axes['N_content'], label='Marion')
    axs[0, 0].legend()
    axs[0, 0].set_xlim(tmin, tmax)
    axs[0, 0].set_ylim(0, 10)
    axs[0, 0].set_ylabel('N content axis (% DM)')

    # N shoot
    axs[0, 1].plot(df_current_axes['t'], df_current_axes['N_content_shoot'], label='current')
    axs[0, 1].plot(df_ref_axes['t'], df_ref_axes['N_content_shoot'], label='Marion')
    axs[0, 1].legend()
    axs[0, 1].set_xlim(tmin, tmax)
    axs[0, 1].set_ylabel('N content shoot (% DM)')

    # N axis
    axs[1, 0].plot(df_current_axes['t'], df_current_axes['sum_N_g'], label='current')
    axs[1, 0].plot(df_ref_axes['t'], df_ref_axes['sum_N_g'], label='Marion')
    axs[1, 0].legend()
    axs[1, 0].set_xlim(tmin, tmax)
    axs[1, 0].set_ylabel('N content axis (g)')
    ax2 = axs[1, 0].twiny()
    ax2.set_xticks(axs[1, 0].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 0].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))

    # N uptake
    axs[1, 1].plot(df_current_organs[df_current_organs['organ'] == 'roots']['t'], df_current_organs[df_current_organs['organ'] == 'roots']['Uptake_Nitrates'], label='current')
    axs[1, 1].plot(df_ref_organs[df_ref_organs['organ'] == 'roots']['t'], df_ref_organs[df_ref_organs['organ'] == 'roots']['Uptake_Nitrates'], label='Marion')
    axs[1, 1].legend()
    axs[1, 1].set_xlim(tmin, tmax)
    axs[1, 1].set_ylabel('Nitrate uptake (µmol)')
    ax2 = axs[1, 1].twiny()
    ax2.set_xticks(axs[1, 1].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 1].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))

    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()


def surface(pdf, df_current_elements, df_ref_elements, meteo_data, tmin, tmax):
    fig, axs = plt.subplots(2, 2)

    # Total green area
    axs[0, 0].plot(df_current_elements['t'].unique(), df_current_elements.groupby('t')['green_area'].sum(), label='current')
    axs[0, 0].plot(df_ref_elements['t'].unique(), df_ref_elements.groupby('t')['green_area'].sum(), label='Marion')
    axs[0, 0].legend()
    axs[0, 0].set_xlim(tmin, tmax)
    axs[0, 0].set_ylabel('Total green area (m²)')

    # Blade green area
    df_current_elements_blade = df_current_elements[df_current_elements.organ == 'blade']
    df_ref_elements_blade = df_ref_elements[df_ref_elements.organ == 'blade']
    axs[0, 1].plot(df_current_elements_blade['t'].unique(), df_current_elements_blade.groupby('t')['green_area'].sum(), label='current')
    axs[0, 1].plot(df_ref_elements_blade['t'].unique(), df_ref_elements_blade.groupby('t')['green_area'].sum(), label='Marion')
    axs[0, 1].legend()
    axs[0, 1].set_xlim(tmin, tmax)
    axs[0, 1].set_ylabel('Blade green area (m²)')

    # Sheath green area
    df_current_elements_sheath = df_current_elements[df_current_elements.organ == 'sheath']
    df_ref_elements_sheath = df_ref_elements[df_ref_elements.organ == 'sheath']
    axs[1, 0].plot(df_current_elements_sheath['t'].unique(), df_current_elements_sheath.groupby('t')['green_area'].sum(), label='current')
    axs[1, 0].plot(df_ref_elements_sheath['t'].unique(), df_ref_elements_sheath.groupby('t')['green_area'].sum(), label='Marion')
    axs[1, 0].legend()
    axs[1, 0].set_xlim(tmin, tmax)
    axs[1, 0].set_ylabel('Sheath green area (m²)')
    ax2 = axs[1, 0].twiny()
    ax2.set_xticks(axs[1, 0].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 0].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))


    # Internode green area
    df_current_elements_internode = df_current_elements[df_current_elements.organ == 'internode']
    df_ref_elements_internode = df_ref_elements[df_ref_elements.organ == 'internode']
    axs[1, 1].plot(df_current_elements_internode['t'].unique(), df_current_elements_internode.groupby('t')['green_area'].sum(), label='current')
    axs[1, 1].plot(df_ref_elements_internode['t'].unique(), df_ref_elements_internode.groupby('t')['green_area'].sum(), label='Marion')
    axs[1, 1].legend()
    axs[1, 1].set_xlim(tmin, tmax)
    axs[1, 1].set_ylabel('Internode green area (m²)')
    ax2 = axs[1, 1].twiny()
    ax2.set_xticks(axs[1, 1].get_xticks())
    ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[1, 1].get_xticks()]['Date']).dt.strftime('%d/%m'))
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 35))

    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()


def include_images(pdf, graphs_dirpath_path, graphs_dirpath_REF_path):
    # graphs_dirpath_to_include = ['leaf_L_hz.PNG', 'Leaf_Lmax.PNG', 'RER_comparison.PNG', 'phyllochron.PNG', 'lamina_Wmax.PNG', 'SSLW.PNG']
    graphs_dirpath_to_include = ['leaf_L_hz.PNG', 'phyllochron.PNG']
    for graph in graphs_dirpath_to_include:
        im = plt.imread(os.path.join(graphs_dirpath_path, graph))
        fig = plt.figure(figsize=(13, 10))
        fig.figimage(im)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()


def leaf_length_mstruct_area(pdf, df_current_hz, df_ref_hz, df_current_elements, df_ref_elements, meteo_data, tmin, tmax):
    # Loop through phytomer
    for phyto_id in df_current_hz['metamer'].unique():
        fig, axs = plt.subplots(3, 3)

        # Length leaf
        axs[0][0].plot(df_current_hz[(df_current_hz.metamer == phyto_id)]['t'], df_current_hz[(df_current_hz.metamer == phyto_id)]['leaf_L'], color='c', label='current')
        if phyto_id in (1,2):
            axs[0][0].plot(df_ref_elements[(df_ref_elements.metamer == phyto_id)]['t'].unique(), df_ref_elements[(df_ref_elements.metamer == phyto_id)].groupby('t')['length'].sum(), color='orange', label='Marion')
        else:
            axs[0][0].plot(df_ref_hz[(df_ref_hz.metamer == phyto_id)]['t'], df_ref_hz[(df_ref_hz.metamer == phyto_id)]['leaf_L'], color='orange', label='Marion')
        axs[0][0].set_xlim(tmin, tmax)
        axs[0][0].set_ylabel('Leaf_L  hz' + ' (m)')
        axs[0][0].set_title('Leaf' + '_' + str(phyto_id))
        axs[0][0].set_xticks([])

        # Length sheath
        axs[0][1].plot(df_current_elements[(df_current_elements.metamer == phyto_id) & (df_current_elements.organ == 'sheath')]['t'].unique(), df_current_elements[(df_current_elements.metamer == phyto_id) & (df_current_elements.organ == 'sheath')].groupby('t')['length'].sum(), color='c', label='current')
        axs[0][1].plot(df_ref_elements[(df_ref_elements.metamer == phyto_id) & (df_ref_elements.organ == 'sheath')]['t'].unique(), df_ref_elements[(df_ref_elements.metamer == phyto_id) & (df_ref_elements.organ == 'sheath')].groupby('t')['length'].sum(), color='orange', label='Marion')
        axs[0][1].set_xlim(tmin, tmax)
        axs[0][1].set_ylabel('Sheath_L elt' + ' (m)')
        axs[0][1].set_xticks([])

        # Mstruct hz
        if phyto_id in (1, 2):
            axs[1][0].plot(df_current_elements[(df_current_elements.metamer == phyto_id) & (df_current_elements.element == 'LeafElement1')]['t'].unique(), df_current_elements[(df_current_elements.metamer == phyto_id) & (df_current_elements.element == 'LeafElement1')]['mstruct'], color='c', label='current')
            axs[1][0].plot(df_ref_elements[(df_ref_elements.metamer == phyto_id)]['t'].unique(), df_ref_elements[(df_ref_elements.metamer == phyto_id) & (df_ref_elements.element == 'LeafElement1')]['mstruct'],
                        color='orange', label='Marion')
        else:
            axs[1][0].plot(df_current_hz[(df_current_hz.metamer == phyto_id)]['t'], df_current_hz[(df_current_hz.metamer == phyto_id)]['mstruct'], color='c', label='current')
            axs[1][0].plot(df_ref_hz[(df_ref_hz.metamer == phyto_id)]['t'], df_ref_hz[(df_ref_hz.metamer == phyto_id)]['mstruct'], color='orange', label='Marion')
        axs[1][0].set_xlim(tmin, tmax)
        axs[1][0].set_ylabel('mstruct hz' + ' (g)')
        axs[1][0].set_xticks([])

        # mstruct lamina
        if phyto_id == 0: continue
        axs[1][1].plot(df_current_elements[(df_current_elements.metamer == phyto_id) & (df_current_elements.organ == 'blade')]['t'].unique(), df_current_elements[(df_current_elements.metamer == phyto_id) & (df_current_elements.organ == 'blade')].groupby('t')['mstruct'].sum(), color='c', label='current')
        axs[1][1].plot(df_ref_elements[(df_ref_elements.metamer == phyto_id) & (df_ref_elements.organ == 'blade')]['t'].unique(), df_ref_elements[(df_ref_elements.metamer == phyto_id) & (df_ref_elements.organ == 'blade')].groupby('t')['mstruct'].sum(), color='orange', label='Marion')
        axs[1][1].set_xlim(tmin, tmax)
        axs[1][1].set_ylabel('Lam mstruct elt' + ' (g)')
        axs[1][1].set_xticks([])

        # mstruct sheath
        axs[2][0].plot(df_current_elements[(df_current_elements.metamer == phyto_id) & (df_current_elements.organ == 'sheath')]['t'].unique(), df_current_elements[(df_current_elements.metamer == phyto_id) & (df_current_elements.organ == 'sheath')].groupby('t')['mstruct'].sum(), color='c', label='current')
        axs[2][0].plot(df_ref_elements[(df_ref_elements.metamer == phyto_id) & (df_ref_elements.organ == 'sheath')]['t'].unique(), df_ref_elements[(df_ref_elements.metamer == phyto_id) & (df_ref_elements.organ == 'sheath')].groupby('t')['mstruct'].sum(), color='orange', label='Marion')
        axs[2][0].set_xlim(tmin, tmax)
        axs[2][0].set_ylabel('Sheath mstruct' + ' (g)')

        ax2 = axs[2][0].twiny()
        ax2.set_xticks(axs[2][0].get_xticks())
        ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[2][0].get_xticks()]['Date']).dt.strftime('%d/%m'))
        ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
        ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
        ax2.spines['bottom'].set_position(('outward', 35))

        # Green area lamina
        current_blade_green_area = df_current_elements[(df_current_elements.metamer == phyto_id) & (df_current_elements.element == 'LeafElement1')]
        Marion_blade_green_area = df_ref_elements[(df_ref_elements.metamer == phyto_id) & (df_ref_elements.element == 'LeafElement1')]
        axs[2][1].plot(current_blade_green_area['t'], current_blade_green_area['green_area'], color='c', label='current')
        axs[2][1].plot(Marion_blade_green_area['t'], Marion_blade_green_area['green_area'], color='orange', label='Marion')
        axs[2][1].set_xlim(tmin, tmax)
        axs[2][1].set_ylabel('Lamina GA' + ' (m2)')

        ax2 = axs[2][1].twiny()
        ax2.set_xticks(axs[2][1].get_xticks())
        ax2.set_xticklabels(pd.to_datetime(meteo_data.loc[axs[2][1].get_xticks()]['Date']).dt.strftime('%d/%m'))
        ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
        ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
        ax2.spines['bottom'].set_position(('outward', 35))

        # Internode length
        current_internode_L = df_current_hz[df_current_hz.metamer == phyto_id]['internode_L']
        Marion_internode_L = df_ref_hz[df_ref_hz.metamer == phyto_id]['internode_L']
        axs[0][2].plot(df_current_hz[(df_current_hz.metamer == phyto_id)]['t'], current_internode_L, color='c', label='current')
        axs[0][2].plot(df_ref_hz[(df_ref_hz.metamer == phyto_id)]['t'], Marion_internode_L, color='orange', label='Marion')
        axs[0][2].set_xlim(tmin, tmax)
        axs[0][2].set_ylabel('Internode_L hz (m)')
        axs[0][2].legend(loc='upper center', bbox_to_anchor=(0.25, 1.5), ncol=2, fontsize="8")

        plt.tight_layout()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

def leaf_emergence(pdf, df_current_hz, df_ref_hz, meteo_data):
    dict_current, dict_marion = {'phyto_id': [], 't_emergence': []}, {'phyto_id': [], 't_emergence_marion': []}
    for phyto_id in df_current_hz['metamer'].unique():
        if not df_current_hz[(df_current_hz.metamer == phyto_id)]['leaf_is_emerged'].any():
            continue
        dict_current['phyto_id'].append(phyto_id)
        dict_current['t_emergence'].append(df_current_hz[(df_current_hz.metamer == phyto_id) & (df_current_hz.leaf_is_emerged == True)]['t'].iloc[0])
    for phyto_id in df_ref_hz['metamer'].unique():
        if phyto_id == 3 or not df_ref_hz[(df_ref_hz.metamer == phyto_id)]['leaf_is_emerged'].any():
            continue
        dict_marion['phyto_id'].append(phyto_id)
        dict_marion['t_emergence_marion'].append(df_ref_hz[(df_ref_hz.metamer == phyto_id) & (df_ref_hz.leaf_is_emerged == True)]['t'].iloc[0])

    df_current = pd.DataFrame.from_dict(dict_current)
    df_marion = pd.DataFrame.from_dict(dict_marion)
    df_merged = df_current.merge(df_marion, on='phyto_id', how='left')
    axis = df_merged.plot(kind='bar')
    axis.set_xlabel('N° de feuille')
    axis.set_ylabel('Temps emergence (hour)')

    ax2 = axis.twinx()
    ax2.set_yticks(axis.get_yticks())
    ax2.set_yticklabels(pd.to_datetime(meteo_data.loc[axis.get_yticks()]['Date']).dt.strftime('%d/%m'))
    ax2.yaxis.set_ticks_position('left')  # set the position of the second x-axis to bottom
    ax2.yaxis.set_label_position('left')  # set the position of the second x-axis to bottom
    ax2.spines['left'].set_position(('outward', 50))

    fig = axis.get_figure()

    plt.tight_layout()
    pdf.savefig(fig)  # saves the current figure into a pdf page
    plt.close()

def plastochrone(pdf, df_current_axes, df_marion_SAMS):
    fig, axis = plt.subplots()

    axis.plot(df_current_axes['t'], df_current_axes['nb_leaves'], label='current')
    axis.plot(df_marion_SAMS['t'], df_marion_SAMS['nb_leaves'], label='current')

    axis.set_xlabel('Time (day)')
    axis.set_ylabel('Number of leaves on MS')
    axis.legend()

    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()


def compare_shoot_outputs(reference_dirpath, newsimu_dirpath, meteo_data_dirpath):

    meteo_data = pd.read_csv(meteo_data_dirpath, index_col='t')

    initial_date = pd.to_datetime("01/11/2000")
    
    # Conversion step from hours 
    meteo_data['Date'] = pd.to_datetime(meteo_data.index.values*1e9*3600 + int(initial_date.timestamp())*1e9)
    meteo_data['Date'] = meteo_data["Date"].dt.strftime('%d/%m/%Y')

    # New simulation Path
    graphs_dirpath = os.path.join(newsimu_dirpath, 'plots')
    newsimu_postprocessing_dirpath = os.path.join(newsimu_dirpath, "postprocessing")

    # Path reference
    refs_graphs_dirpath = os.path.join(reference_dirpath, 'plots')
    reference_postprocessing_dirpath = reference_dirpath

    # Axes
    df_current_axes = pd.read_csv(os.path.join(newsimu_postprocessing_dirpath, 'axes_postprocessing.csv'))
    df_current_axes = df_current_axes[df_current_axes['axis'] == 'MS']
    df_ref_axes = pd.read_csv(os.path.join(reference_postprocessing_dirpath, 'axes_postprocessing.csv'))
    df_ref_axes = df_ref_axes[df_ref_axes['axis'] == 'MS']
    df_ref_axes['t'] = df_ref_axes['t'] + delta_t_simuls

    # SAMs
    # df_marion_SAMS = pd.read_csv(os.path.join(dirpath_marion, 'outputs', 'SAM_states.csv'))
    # df_marion_SAMS = df_marion_SAMS[df_marion_SAMS['axis'] == 'MS']
    # df_marion_SAMS['t'] = df_marion_SAMS['t'] + delta_t_simuls

    # Organs
    df_current_organs = pd.read_csv(os.path.join(newsimu_postprocessing_dirpath, 'organs_postprocessing.csv'))
    df_current_organs = df_current_organs[df_current_organs['axis'] == 'MS']
    df_ref_organs = pd.read_csv(os.path.join(reference_postprocessing_dirpath, 'organs_postprocessing.csv'))
    df_ref_organs = df_ref_organs[df_ref_organs['axis'] == 'MS']
    df_ref_organs['t'] = df_ref_organs['t'] + delta_t_simuls

    # Elements
    df_current_elements = pd.read_csv(os.path.join(newsimu_postprocessing_dirpath, 'elements_postprocessing.csv'))
    df_current_elements = df_current_elements[df_current_elements['axis'] == 'MS']
    df_ref_elements = pd.read_csv(os.path.join(reference_postprocessing_dirpath, 'elements_postprocessing.csv'))
    df_ref_elements = df_ref_elements[df_ref_elements['axis'] == 'MS']
    df_ref_elements['t'] = df_ref_elements['t'] + delta_t_simuls

    # HZ
    df_current_hz = pd.read_csv(os.path.join(newsimu_postprocessing_dirpath, 'hiddenzones_postprocessing.csv'))
    df_current_hz = df_current_hz[df_current_hz['axis'] == 'MS']
    df_ref_hz = pd.read_csv(os.path.join(reference_postprocessing_dirpath, 'hiddenzones_postprocessing.csv'))
    df_ref_hz = df_ref_hz[df_ref_hz['axis'] == 'MS']
    df_ref_hz['t'] = df_ref_hz['t'] + delta_t_simuls

    tmin = df_current_axes.t.min()
    tmax = df_current_axes.t.max()

    # plot graphs_dirpath
    with PdfPages(os.path.join(newsimu_dirpath, 'Comparison_Marion.pdf')) as pdf:
        # phloem
        phloem(pdf, df_current_organs, df_ref_organs, meteo_data, tmin, tmax)

        # Photosynthesis
        photosynthesis(pdf, df_current_axes, df_ref_axes)

        # roots
        roots(pdf, df_current_organs, df_ref_organs, meteo_data, tmin, tmax)

        # dry mass & shoot : root
        dry_mass(pdf, df_current_axes, df_ref_axes, df_current_organs, df_ref_organs, meteo_data, tmin, tmax)

        # N mass
        N_mass(pdf, df_current_axes, df_ref_axes, df_current_organs, df_ref_organs, meteo_data, tmin, tmax)

        # Surfaces
        surface(pdf, df_current_elements, df_ref_elements, meteo_data, tmin, tmax)
        include_images(pdf, graphs_dirpath, refs_graphs_dirpath)

        # Leaf length & mstruct
        leaf_length_mstruct_area(pdf, df_current_hz, df_ref_hz, df_current_elements, df_ref_elements, meteo_data, tmin, tmax)

        # Leaf emergence date
        leaf_emergence(pdf, df_current_hz, df_ref_hz, meteo_data)

        # Plastochron
        # plastochrone(df_current_axes, df_marion_SAMS)