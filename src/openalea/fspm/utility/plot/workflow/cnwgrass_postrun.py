import os
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from openalea.cnwgrass.integration import cnmetabolism_facade
from openalea.cnwgrass.integration import hydraulics_facade
from openalea.cnwgrass.morphogenesis.parameters import Parameters as MorphogenesisParameters
from openalea.cnwgrass.integration import tools as integration_tools

delta_t_simuls = 0


AXES_INDEX_COLUMNS = ['t', 'plant', 'axis']
ELEMENTS_INDEX_COLUMNS = ['t', 'plant', 'axis', 'metamer', 'organ', 'element']
HIDDENZONES_INDEX_COLUMNS = ['t', 'plant', 'axis', 'metamer']
ORGANS_INDEX_COLUMNS = ['t', 'plant', 'axis', 'organ']
SOILS_INDEX_COLUMNS = ['t', 'plant', 'axis']

# Name of the CSV files which contain the outputs of the model
AXES_OUTPUTS_FILENAME = 'axes_outputs.csv'
ORGANS_OUTPUTS_FILENAME = 'organs_outputs.csv'
HIDDENZONES_OUTPUTS_FILENAME = 'hiddenzones_outputs.csv'
ELEMENTS_OUTPUTS_FILENAME = 'elements_outputs.csv'
SOILS_OUTPUTS_FILENAME = 'soil_outputs.csv'

# Name of the CSV files which contain the postprocessing outputs of the model
AXES_POSTPROCESSING_FILENAME = 'axes_postprocessing.csv'
ORGANS_POSTPROCESSING_FILENAME = 'organs_postprocessing.csv'
HIDDENZONES_POSTPROCESSING_FILENAME = 'hiddenzones_postprocessing.csv'
ELEMENTS_POSTPROCESSING_FILENAME = 'elements_postprocessing.csv'
SOILS_POSTPROCESSING_FILENAME = 'soils_postprocessing.csv'

OUTPUTS_PRECISION = 8


def cnwgrass_postprocessing(csv_dirpath, hydraulics: bool=True):
    print("[cnwgrass_postprocessing] CNW-Grass postprocessing...")
    if not os.path.isdir(csv_dirpath):
        os.mkdir(csv_dirpath)

    # --- Generate graphs from postprocessing files
    plt.ioff()
    delta_t = 3600
    df_elt = pd.read_csv(os.path.join(csv_dirpath, ELEMENTS_OUTPUTS_FILENAME))
    df_org = pd.read_csv(os.path.join(csv_dirpath, ORGANS_OUTPUTS_FILENAME))
    df_hz = pd.read_csv(os.path.join(csv_dirpath, HIDDENZONES_OUTPUTS_FILENAME))
    df_SAM = pd.read_csv(os.path.join(csv_dirpath, AXES_OUTPUTS_FILENAME))
    df_soil = pd.read_csv(os.path.join(csv_dirpath, SOILS_OUTPUTS_FILENAME))

    postprocessing = cnmetabolism_facade.CNMetabolismFacade.postprocessing(
                                axes_outputs_df=df_SAM,
                                hiddenzone_outputs_df=df_hz,
                                organs_outputs_df=df_org,
                                elements_outputs_df=df_elt,
                                soils_outputs_df=df_soil,
                                delta_t=delta_t)

    if hydraulics:
        turgor_postprocessing = hydraulics_facade.hydraulicsFacade.postprocessing(axes_outputs_df=df_SAM,
                                                                                hiddenzone_outputs_df=df_hz,
                                                                                elements_outputs_df=df_elt,
                                                                                organs_outputs_df=df_org,
                                                                                soils_outputs_df=df_soil,
                                                                                delta_t=delta_t)

        # Merge with cnmetabolism postprocessing
        mapping_scales = [('axes', AXES_INDEX_COLUMNS),
            ('elements', ELEMENTS_INDEX_COLUMNS),
            ('hiddenzones', HIDDENZONES_INDEX_COLUMNS),
            ('organs', ORGANS_INDEX_COLUMNS),
            ('soils', SOILS_INDEX_COLUMNS)]

        for scale, index_cols in mapping_scales:
            df_cnmetabolism = postprocessing.get(scale)
            df_turgor = turgor_postprocessing.get(scale)

            turgor_exclusive_cols = index_cols + [col for col in df_turgor.columns if col not in df_cnmetabolism.columns]
            df_turgor_filtered = df_turgor[turgor_exclusive_cols]

            # Left merge
            postprocessing[scale] = pd.merge(df_cnmetabolism, df_turgor_filtered, on=index_cols, how='left')
    

    save_postprocessing = True
    if save_postprocessing:
        postprocessing_dirpath = os.path.join(csv_dirpath, "postprocessing")
        if os.path.isdir(postprocessing_dirpath):
            shutil.rmtree(postprocessing_dirpath)
        os.mkdir(postprocessing_dirpath)

        for postprocessing_file_basename, postprocessing_filename, index_columns in (('axes', AXES_POSTPROCESSING_FILENAME, AXES_INDEX_COLUMNS),
                                                                                     ('hiddenzones', HIDDENZONES_POSTPROCESSING_FILENAME, HIDDENZONES_INDEX_COLUMNS),
                                                                                     ('organs', ORGANS_POSTPROCESSING_FILENAME, ORGANS_INDEX_COLUMNS),
                                                                                     ('elements', ELEMENTS_POSTPROCESSING_FILENAME, ELEMENTS_INDEX_COLUMNS),
                                                                                     ('soils', SOILS_POSTPROCESSING_FILENAME, SOILS_INDEX_COLUMNS)):
            postprocessing_filepath = os.path.join(postprocessing_dirpath, postprocessing_filename)
            postprocessing_df = postprocessing[postprocessing_file_basename]
            postprocessing_df.rename({'level_0': 't'}, axis=1, inplace=True)
            postprocessing_df = postprocessing_df.reindex(index_columns + postprocessing_df.columns.difference(index_columns).tolist(), axis=1, copy=False)
            postprocessing_df.to_csv(postprocessing_filepath, na_rep='NA', index=False, float_format='%.{}f'.format(OUTPUTS_PRECISION))
    
    print("[cnwgrass_postprocessing] CNW-Grass postprocessing: DONE")

def cnwgrass_plots(csv_dirpath, plant_density = 250, inputs_dirpath: str = 'inputs', hydraulics: bool=True):
    print("[cnwgrass_plots] Opening CSV files...")
    postprocessing_dirpath = os.path.join(csv_dirpath, "postprocessing")

    postprocessing = {}

    for postprocessing_filename in (AXES_POSTPROCESSING_FILENAME,
                                    ORGANS_POSTPROCESSING_FILENAME,
                                    HIDDENZONES_POSTPROCESSING_FILENAME,
                                    ELEMENTS_POSTPROCESSING_FILENAME,
                                    SOILS_POSTPROCESSING_FILENAME):
        postprocessing_filepath = os.path.join(postprocessing_dirpath, postprocessing_filename)
        postprocessing_df = pd.read_csv(postprocessing_filepath)
        postprocessing_file_basename = postprocessing_filename.split('_')[0]
        postprocessing[postprocessing_file_basename] = postprocessing_df

    outputs_df_dict = {}

    for outputs_filename in (AXES_OUTPUTS_FILENAME,
                                ORGANS_OUTPUTS_FILENAME,
                                HIDDENZONES_OUTPUTS_FILENAME,
                                ELEMENTS_OUTPUTS_FILENAME,
                                SOILS_OUTPUTS_FILENAME):
        outputs_filepath = os.path.join(csv_dirpath, outputs_filename)
        outputs_df = pd.read_csv(outputs_filepath, dtype={'is_over': str, 'is_growing': str})
        outputs_file_basename = outputs_filename.split('.')[0]
        outputs_df_dict[outputs_file_basename] = outputs_df

        # Assert states_filepaths were not opened during simulation run meaning that other filenames were saved
        tmp_filename = 'ACTUAL_{}.csv'.format(outputs_file_basename)
        tmp_path = os.path.join(csv_dirpath, tmp_filename)
        assert not os.path.isfile(tmp_path), \
            "File {} was saved because {} was opened during simulation run. Rename it before running postprocessing".format(
                tmp_filename, outputs_file_basename)

    plot_path = os.path.join(csv_dirpath, "plots")

    if os.path.isdir(plot_path):
        shutil.rmtree(plot_path)
    os.mkdir(plot_path)

    meteo = pd.read_csv(os.path.join(inputs_dirpath, "meteo_Ljutovac2002.csv"), index_col='t')

    print("[cnwgrass_plots] Opening CSV files: DONE")
    print("[cnwgrass_plots] Producing graphs...")

    # --- Generate graphs from postprocessing files
    plt.ioff()

    cnmetabolism_facade.CNMetabolismFacade.graphs(axes_postprocessing_df=postprocessing['axes'],
                                                    hiddenzones_postprocessing_df=postprocessing['hiddenzones'],
                                                    organs_postprocessing_df=postprocessing['organs'],
                                                    elements_postprocessing_df=postprocessing['elements'],
                                                    soils_postprocessing_df=postprocessing['soils'],
                                                    meteo_data=meteo, graphs_dirpath=plot_path)

    if hydraulics:
        hydraulics_facade.hydraulicsFacade.graphs(axes_postprocessing_df=postprocessing['axes'],
                                                    hiddenzones_postprocessing_df=postprocessing['hiddenzones'],
                                                    organs_postprocessing_df=postprocessing['organs'],
                                                    elements_postprocessing_df=postprocessing['elements'],
                                                    soils_postprocessing_df=postprocessing['soils'],
                                                    meteo_data=meteo, graphs_dirpath=plot_path)
    # --- Additional graphs
    data_obs = pd.read_csv(os.path.join(inputs_dirpath, 'Ljutovac2002.csv'))
    RERmax_items = MorphogenesisParameters().RERmax.items()
    integration_tools.additional_graphs(outputs_df_dict['axes_outputs'], outputs_df_dict['hiddenzones_outputs'], outputs_df_dict['elements_outputs'],
                                        postprocessing['axes'], postprocessing['hiddenzones'], postprocessing['elements'], postprocessing['organs'],
                                        plant_density, RERmax_items, plot_path, data_obs)

    print("[cnwgrass_plots] Producing graphs: DONE")


def compare_cnwgrass_outputs(reference_dirpath, newsimu_dirpath, meteo_data_dirpath):
    print("[cnwgrass_postprocessing] Comparing CNW-Grass outputs...")

    meteo_data = pd.read_csv(meteo_data_dirpath, index_col='t')

    initial_date = pd.to_datetime("17/12/1998", dayfirst=True)
    
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

    # C_allocation(dirpath=newsimu_dirpath, df_org=df_current_organs, df_org_ref=df_ref_organs, 
    #              df_axe=df_current_axes, df_axe_ref=df_current_axes, df_elt=df_current_elements, df_elt_ref=df_ref_elements)
    

    tmin = df_current_axes.t.min()
    tmax = df_current_axes.t.max()

    # dry_mass(None, df_current_axes, df_ref_axes, df_current_organs, df_ref_organs, meteo_data, tmin, tmax, dirpath=newsimu_dirpath)
    
    # plot graphs_dirpath
    with PdfPages(os.path.join(newsimu_dirpath, 'Comparison_Marion.pdf')) as pdf:

        print("Trying to create output pdf")
        # phloem
        phloem(pdf, df_current_organs, df_ref_organs, meteo_data, tmin, tmax)

        # Photosynthesis
        photosynthesis(pdf, df_current_axes, df_ref_axes)

        # roots
        roots(pdf, df_current_organs, df_ref_organs, meteo_data, tmin, tmax)

        # dry mass & shoot : root
        dry_mass(pdf, df_current_axes, df_ref_axes, df_current_organs, df_ref_organs, meteo_data, tmin, tmax, dirpath=newsimu_dirpath)

        # N mass
        N_mass(pdf, df_current_axes, df_ref_axes, df_current_organs, df_ref_organs, meteo_data, tmin, tmax)

        # Surfaces
        surface(pdf, df_current_elements, df_ref_elements, meteo_data, tmin, tmax)
        include_images(pdf, graphs_dirpath, refs_graphs_dirpath)

        # Leaf length & mstruct
        leaf_length_mstruct_area(pdf, df_current_hz, df_ref_hz, df_current_elements, df_ref_elements, meteo_data, tmin, tmax)

        # Leaf emergence date
        leaf_emergence(pdf, df_current_hz, df_ref_hz, meteo_data)

        C_allocation(pdf=pdf, dirpath=newsimu_dirpath, df_org=df_current_organs, df_org_ref=df_ref_organs, 
                 df_axe=df_current_axes, df_axe_ref=df_ref_axes, df_elt=df_current_elements, df_elt_ref=df_ref_elements)
        
        # Plastochron
        # plastochrone(df_current_axes, df_marion_SAMS)

    print("[cnwgrass_postprocessing] Comparing CNW-Grass outputs: DONE")



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

def dry_mass(pdf, df_current_axes, df_ref_axes, df_current_organs, df_ref_organs, meteo_data, tmin, tmax, dirpath):
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
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
    else:
        fig.savefig(os.path.join(dirpath, 'dry_mass.PNG'), dpi=720, format='PNG', bbox_inches='tight')
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
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
    else:
        fig.savefig(os.path.join(dirpath, 'shoot_root.PNG'), dpi=720, format='PNG', bbox_inches='tight')
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

def C_allocation(dirpath, df_org, df_org_ref, df_axe, df_axe_ref, df_elt, df_elt_ref, pdf=None):
    # 4) Total C production vs. Root C allcoation
    # df_org = postprocessing_df_dict[organs_postprocessing_file_basename]
    df_roots = df_org[df_org['organ'] == 'roots'].copy()
    df_roots['day'] = df_roots['t'] // 24 + 1
    df_roots['Unloading_Sucrose_tot'] = df_roots['Unloading_Sucrose'] * df_roots['mstruct']
    Unloading_Sucrose_tot = df_roots.groupby(['day'])['Unloading_Sucrose_tot'].agg('sum')
    days = df_roots['day'].unique()

    df_roots_ref = df_org_ref[df_org_ref['organ'] == 'roots'].copy()
    df_roots_ref['day'] = df_roots_ref['t'] // 24 + 1
    df_roots_ref['Unloading_Sucrose_tot'] = df_roots_ref['Unloading_Sucrose'] * df_roots_ref['mstruct']
    Unloading_Sucrose_tot_ref = df_roots_ref.groupby(['day'])['Unloading_Sucrose_tot'].agg('sum')
    days_ref = df_roots_ref['day'].unique()

    # df_axe = postprocessing_df_dict[axes_postprocessing_file_basename]
    df_axe['day'] = df_axe['t'] // 24 + 1
    Total_Photosynthesis = df_axe.groupby(['day'])['Tillers_Photosynthesis'].agg('sum')
    
    df_axe_ref['day'] = df_axe_ref['t'] // 24 + 1
    Total_Photosynthesis_ref = df_axe_ref.groupby(['day'])['Tillers_Photosynthesis'].agg('sum')

    # df_elt = postprocessing_df_dict[elements_postprocessing_file_basename]
    df_elt['day'] = df_elt['t'] // 24 + 1
    df_elt['sum_respi_tillers'] = df_elt['sum_respi'] * df_elt['nb_replications']
    Shoot_respiration = df_elt.groupby(['day'])['sum_respi_tillers'].agg('sum')
    Net_Photosynthesis = Total_Photosynthesis - Shoot_respiration

    share_net_roots_live = Unloading_Sucrose_tot / Net_Photosynthesis * 100

    df_elt_ref['day'] = df_elt_ref['t'] // 24 + 1
    df_elt_ref['sum_respi_tillers'] = df_elt_ref['sum_respi'] * df_elt_ref['nb_replications']
    Shoot_respiration_ref = df_elt_ref.groupby(['day'])['sum_respi_tillers'].agg('sum')
    Net_Photosynthesis_ref = Total_Photosynthesis_ref - Shoot_respiration_ref

    share_net_roots_live_ref = Unloading_Sucrose_tot_ref / Net_Photosynthesis_ref * 100

    fig, ax = plt.subplots()
    line1 = ax.plot(days, Unloading_Sucrose_tot, label=u'Current')
    line2 = ax.plot(days, Unloading_Sucrose_tot_ref, label=u'Marion')
    lines = line1 + line2
    labs = [line.get_label() for line in lines]
    ax.legend(lines, labs, loc='center left', prop={'size': 10}, framealpha=0.5, bbox_to_anchor=(1, 0.815), borderaxespad=0.)
    ax.set_xlabel('Days')
    ax.set_ylabel(u'C (µmol C.day$^{-1}$ )')
    ax.set_title('C allocation to roots')
    if pdf is not None:
        pdf.savefig()
    else:
        fig.savefig(os.path.join(dirpath, 'C_allocation_to_roots.PNG'), dpi=200, format='PNG', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    line1 = ax.plot(days, Net_Photosynthesis, label=u'Current')
    line2 = ax.plot(days, Net_Photosynthesis_ref, label=u'Marion')
    lines = line1 + line2
    labs = [line.get_label() for line in lines]
    ax.legend(lines, labs, loc='center left', prop={'size': 10}, framealpha=0.5, bbox_to_anchor=(1, 0.815), borderaxespad=0.)
    ax.set_xlabel('Days')
    ax.set_ylabel(u'C (µmol C.day$^{-1}$ )')
    ax.set_title('Net_Photosynthesis')
    if pdf is not None:
        pdf.savefig()
    else:
        fig.savefig(os.path.join(dirpath, 'C_allocation_net_photosynthesis.PNG'), dpi=200, format='PNG', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    line1 = ax.plot(days, share_net_roots_live, label=u'Current')
    line2 = ax.plot(days, share_net_roots_live_ref, label=u'Marion')
    lines = line1 + line2
    labs = [line.get_label() for line in lines]
    ax.legend(lines, labs, loc='center left', prop={'size': 10}, framealpha=0.5, bbox_to_anchor=(1, 0.815), borderaxespad=0.)
    ax.set_xlabel('Days')
    ax.set_ylabel(u'Ratio (%)')
    ax.set_title('Net C Shoot production sent to roots (%)')
    if pdf is not None:
        pdf.savefig()
    else:
        fig.savefig(os.path.join(dirpath, 'C_allocation_net_C_shoot_production_to_roots.PNG'), dpi=200, format='PNG', bbox_inches='tight')
    plt.close()
