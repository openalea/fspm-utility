from initialize import mtg_from_expe_rsml
from log.visualize import plot_mtg_alt, post_compress_gltf
import pyvista as pv
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pickle
import pandas as pd 
from openalea.mtg.traversal import pre_order2, post_order2
from openalea.mtg import MTG



def spread_mtg_ages(g):
    root = 1
    props = g.properties()
    if "length_growth_rate" not in props:
        props.setdefault("length_growth_rate", {})

    processed_vids = []
    max_date = 42.
    range_vids = []
    wait_for_parents = {}
    emergence_delay = 0.1

    for vid in post_order2(g, root):
        if vid not in processed_vids:
            axis = g.Axis(vid)
            processed_vids += axis
            previous_date = None
            for seg in sorted(axis, reverse=True):
                if seg not in props["Number"]:
                    if previous_date:
                        props["Number"][seg] = previous_date
                    else:
                        props["Number"][seg] = max_date
                    range_vids.append(seg)
                else:
                    props["Number"][seg] = float(props["Number"][seg])
                    
                    if previous_date:
                        time_span = previous_date - props["Number"][seg]
                    else:
                        time_span = max_date - props["Number"][seg]
                    
                    previous_date = props["Number"][seg]

                    range_length = sum([props["length"][v] for v in range_vids])

                    if time_span == 0:
                        growth_rate = 0
                    else:
                        growth_rate = range_length / (time_span * 24 * 3600)

                    for v in range_vids:
                        props["length_growth_rate"][v] = growth_rate

                    range_vids = [seg]

            if previous_date:
                wait_for_parents[previous_date] = range_vids
            else:
                wait_for_parents[max_date] = range_vids
    
    for date, vids in wait_for_parents.items():
        insertion_id = min(vids)
        parent = g.parent(insertion_id)
        if parent:
            time_span = date - (props["Number"][parent] + emergence_delay / props["length_growth_rate"][parent])
        else:
            time_span = date
        
        range_length = sum([props["length"][v] for v in range_vids])

        if time_span == 0:
            growth_rate = 0
        else:
            growth_rate = range_length / (time_span * 24 * 3600)
        growth_rate = range_length / (time_span * 24 * 3600)

        for v in range_vids:
            props["length_growth_rate"][v] = growth_rate

def estimate_growth_costs(g, RTD, struct_mass_C_content):

    vertices = g.vertices(g.max_scale())
    props = g.properties()

    if "C_consumption_by_growth" not in props:
        props.setdefault("C_consumption_by_growth", {})

    for vid in vertices:
        n = g.node(vid)
        n.C_consumption_by_growth = n.length_growth_rate * np.pi * (n.radius ** 2) * RTD * struct_mass_C_content

def estimate_struct_mass(g, RTD):
    vertices = g.vertices(g.max_scale())
    props = g.properties()

    if "struct_mass" not in props:
        props.setdefault("struct_mass", {})
    
    for vid in vertices:
        n = g.node(vid)
        n.struct_mass = n.length * np.pi * (n.radius ** 2) * RTD


def mtg_subset_from_array(g, vid_list):

    sub_g = MTG()

    vid_mapping = {}

    for vid in vid_list:
        properties = g.get_vertex_property(vid)
        new_vid = sub_g.add_component(complex_id=0, **properties)
        vid_mapping[vid] = new_vid

    for vid in vid_list:
        parent = g.parent(vid)
        if parent in vid_list:
            sub_g.add_child(parent=vid_mapping[parent], child=vid_mapping[vid])

    return sub_g

def Archisimple_traits_from_mtg(g, vid_filter):
    props = g.properties()
    vertices = vid_filter
    traits = {}

    for order in np.unique(list(props["order"].values())):
        mean = np.mean([2*props["radius"][vid] for vid in vertices if props["order"][vid] == order])
        if not np.isnan(mean):
            traits[f"mean_diameter_order_{order}"] = mean
    
    traits["RDM"] = traits["mean_diameter_order_2"] / traits["mean_diameter_order_1"]

    traits["Di"] = [2 * props["radius"][vid] for vid in vertices if props["label"][vid] == "Apex" and props["order"][vid] == 1][0]

    traits["Dmin"] = 2 * min(props["radius"].values())

    # IPD
    base_element = min(vid_filter)
    traits["IPD"] = {}
    previous_ramification_vid = None
    if "distance_from_base" not in props:
        props.setdefault("distance_from_base", {})
    props["distance_from_base"][1] = 0

    for vid in sorted(g.Axis(base_element)):
        parent = g.parent(vid)
        if parent:
            props["distance_from_base"][vid] = props["distance_from_base"][parent] + props["length"][parent]
        children = g.children(vid)
        ramifications = [v for v in children if props["edge_type"][v] == "+"]
        if len(ramifications) > 0:
            for ramif in ramifications:
                # Then this is a ramification
                if previous_ramification_vid:
                    local_ipd = props["distance_from_base"][vid] - props["distance_from_base"][previous_ramification_vid]
                    ipd_location = (props["distance_from_base"][vid] + props["distance_from_base"][previous_ramification_vid]) / 2
                    traits["IPD"][ipd_location] = local_ipd

                previous_ramification_vid = vid

    traits["mean_IPD"] = np.mean(list(traits["IPD"].values()))

    # EL
    y = np.array([props["length_growth_rate"][v] for v in vid_filter])
    x = np.array([props["radius"][v] for v in vid_filter]) * 2

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    traits["EL"] = slope

    return traits

usual_clim = dict(
    length_growth_rate=[1e-10, 5e-5],
    C_consumption_by_growth=[1e-11, 1e-10],
    massic_C_consumption=[1e-7, 1e-5]
)

if __name__ == "__main__":

    input_folder = "test/inputs"

    # output_folder = "outputs"
    output_folder = "test/outputs"

    input_translator = "R14_global_root_data.csv"

    from_rsml = False

    translator_df = pd.read_csv(os.path.join(input_folder, input_translator))
    translator = dict(zip(translator_df[" root"].values, translator_df[" parent"]))

    if from_rsml:
        input_rsml = os.path.join(input_folder, "verso_grayscale_R14_D44 .rsml")
        g = mtg_from_expe_rsml(input_rsml, length_unit_conversion_factor=1/45625.6849, translator=translator)

        with open(os.path.join(output_folder, "R14_mtg.pckl"), "wb") as f:
            pickle.dump(g, f)

    else:
        input_mtg = os.path.join(output_folder, "R14_mtg.pckl")

        with open(input_mtg, "rb") as f:
            g = pickle.load(f)

    kept_axis = 2
    for child in g.children(1):
        if child == kept_axis:
            vid_filter=list(pre_order2(g, child)) + list(g.Sons(child))

    vertices = g.vertices(scale=g.max_scale())
    props = g.properties()
    spread_mtg_ages(g)

    estimate_growth_costs(g, RTD = 0.1 *1e6, struct_mass_C_content=0.44 / 12.01)

    estimate_struct_mass(g, RTD= 0.1 * 1e6)

    props.setdefault('massic_C_consumption', {})
    for vid in vertices:
        n = g.node(vid)
        n.massic_C_consumption = n.C_consumption_by_growth / n.struct_mass

    plotting = True

    if plotting:
        cmap_property = "length_growth_rate"

        root_system, _, _ = plot_mtg_alt(g, cmap_property=cmap_property, flow_property=False, vid_filter=vid_filter)

        plotter = pv.Plotter(lighting="three lights")

        plotter.add_mesh(root_system, cmap="jet", clim=usual_clim[cmap_property], show_edges=False,
                                                            specular=1., log_scale=True)
        
        plotter.export_gltf(os.path.join(output_folder, f'root_2_{cmap_property}.gltf'))
        # post_compress_gltf(output_folder)

        plotter.show(interactive_update=False)

    traits = Archisimple_traits_from_mtg(g, vid_filter=vid_filter)

    traits["root_label"] = "R14_root_2"

    exportable_traits = {k: v for k, v in traits.items() if isinstance(v, float) or isinstance(v, str)}
    
    df = pd.DataFrame(exportable_traits, index=[0])

    df.set_index('root_label')

    df.to_csv(os.path.join(output_folder, "R14_traits.csv"))

    

