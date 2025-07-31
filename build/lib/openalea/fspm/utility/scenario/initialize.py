import os
import pandas as pd
import numpy as np
import pickle
from SALib.sample import saltelli
from openalea.mtg import MTG
from openalea.mtg.traversal import pre_order2
import xml.etree.ElementTree as ET


class MakeScenarios:

    def from_table(file_path, which=[]):
        instructions = read_table(file_path, index_col="Input")
        input_directory = os.path.dirname(file_path)

        if len(which) > 0:
            scenario_names = which
        else:
            scenario_names = [scenario_name for scenario_name in instructions.columns if scenario_name not in ("Input", "Input_type", "Dedicated_to", "Organ_label", "Explanation", "Type/ Unit", "Reference_value")]

        instructions_parameters = instructions.loc[instructions["Input_type"] == "parameter"]
        targeted_models = list(set(instructions_parameters["Dedicated_to"].values))

        # This accounts for different cases of parameter editing for different models
        subdict_of_parameters = {}
        for name in scenario_names:
            subdict_of_parameters.update({name: {}})
            for model in targeted_models:
                subdict_of_parameters[name].update({model: {}})
                table_extract = instructions_parameters[instructions_parameters["Dedicated_to"] == model]
                label = list(set(table_extract["Organ_label"].values))[0]
                # This accounts for the case where passed parameters to every models have to be encapsulated in a Organ-labelled dict
                if label is not None:
                    subdict_of_parameters[name][model].update(
                        {label: dict(zip(table_extract[name].index.values, table_extract[name].replace({'True': True, 'False': False, 'None': None})))}
                    )
                else:
                    subdict_of_parameters[name][model].update(
                        dict(zip(table_extract[name].index.values, table_extract[name].replace({'True': True, 'False': False, 'None': None})))
                    )

        instructions_table_file = instructions.loc[instructions["Input_type"] == "input_tables"]
        instructions_initial_mtg_file = instructions.loc[instructions["Input_type"] == "input_mtg"]
        scenarios = {name: {
            "parameters": subdict_of_parameters[name],
            "input_tables": {var: read_table(os.path.join(input_directory, str(instructions_table_file[name][var])), index_col="t")[var] 
                             for var in instructions_table_file.index.values if not pd.isna(instructions_table_file[name][var])} if len(instructions_table_file) > 0 else None,
            "input_mtg": {var: read_mtg(os.path.join(input_directory, str(instructions_initial_mtg_file[name][var]))) if not pd.isna(instructions_initial_mtg_file[name][var]) 
                          else None for var in instructions_initial_mtg_file.index.values} if len(instructions_initial_mtg_file) > 0 else None}
                     for name in scenario_names}

        return scenarios

    def from_factorial_plan(file_path, save_scenarios=True, N=10):
        factorial_plan = read_table(file_path, index_col="Input")
        input_directory = os.path.dirname(file_path)
        reference_dirpath = os.path.join(input_directory, "Scenarios_24_05.xlsx")
        reference_scenario = read_table(reference_dirpath, index_col="Input")
        reference_scenario = reference_scenario[["Input_type", "Dedicated_to", "Organ_label", "Explanation", "Type/ Unit", "Reference_value", "Reference_Fischer"]]

        SA_problem = {}
        
        factors = factorial_plan.index.to_list()
        SA_problem["num_vars"] = len(factors)
        SA_problem["names"] = factors
        SA_problem["bounds"] = factorial_plan[["Min", "Max"]].values.tolist()
        
        param_values = saltelli.sample(SA_problem, N=N, calc_second_order=True)
        scenario_names = [f"SA{i}" for i in range(len(param_values))]

        # Now produce the dataframe containing all the scenarios as rows, through an edition of the reference
        SA_scenarios = reference_scenario
        for k in range(len(param_values)):
            edited_scenario = reference_scenario["Reference_Fischer"].to_dict()
            for f in range(len(param_values[k])):
                edited_scenario[factors[f]] = param_values[k][f]
            SA_scenarios[scenario_names[k]] = edited_scenario

        output_filename = os.path.join(input_directory, "Scenarios_SA.xlsx")
        if save_scenarios:
            SA_scenarios.to_excel(output_filename)

        return SA_problem, output_filename, scenario_names



def read_table(file_path, index_col=None):
    if file_path.lower().endswith((".csv", ".xlsx")):
        # Add more types then if necessary
        if file_path.lower().endswith(".xlsx"):
            return pd.read_excel(file_path, index_col=index_col)

        elif file_path.lower().endswith(".csv"):
            return pd.read_csv(file_path, index_col=index_col, sep=";|,", engine="python")
    elif file_path == 'None':
        return None
    else:
        raise TypeError("Only tables are allowed")
    

def read_mtg(file_path):
    """
    General reader for MTG from native mtg file fromat or rsml xml format
    """
    if file_path.endswith(".pckl"):
        with open(file_path, "rb") as f:
            g = pickle.load(f)
    elif file_path.endswith(".rsml"):
        g = mtg_from_rsml(file_path)
    
    else:
        g = None

    return g
    

def mtg_from_expe_rsml(file_path: str, length_unit_conversion_factor: int, translator: dict, diameter_filter_threshold: float = 0.5, correcting_rsml=False):
    """
    param: min_lengt in m
    """

    polylines, properties, functions, annotations, right_seed, left_seed, ids, insertions = read_rsml(file_path)

    for f, v in functions.items():
        functions[f] = [[value * length_unit_conversion_factor if value else None for value in l ] for l in v]

    for k in range(len(annotations)):
        annotations[k]["x"] = annotations[k]["x"] * length_unit_conversion_factor
        annotations[k]["y"] = annotations[k]["y"] * length_unit_conversion_factor

    polylines = [[[i[j] * length_unit_conversion_factor for j in range(2)] for i in k] for k in polylines]
    right_seed = [[i * length_unit_conversion_factor for i in k] for k in right_seed]
    left_seed = [[i * length_unit_conversion_factor for i in k] for k in left_seed]
    insertions = [k * length_unit_conversion_factor for k in insertions]

    # We define the first base element as an empty element:

    right_seed_id = '7b54894e-e4bc-4d7d-a3c7-5b136c08c84e'

    # We create an empty MTG:
    g_right = MTG()
    g_left = MTG()
    polyline_to_mtg_vid = {}

    id_segment = g_right.add_component(g_right.root, label='Segment',
                                 type="Base_of_the_root_system",
                                 x1=right_seed[0][0],
                                 x2=right_seed[1][0],
                                 y1=right_seed[0][1],
                                 y2=right_seed[1][1],
                                 z1=0,
                                 z2=0,
                                 radius1=0.002,
                                 radius2=0.002,
                                 radius=0.002,
                                 length=0,
                                 order=0,
                                 insertion=None
                                 )
    base_segment_right = g_right.node(id_segment)

    id_segment = g_left.add_component(g_left.root, label='Segment',
                                 type="Base_of_the_root_system",
                                 x1=left_seed[0][0],
                                 x2=left_seed[1][0],
                                 y1=left_seed[0][1],
                                 y2=left_seed[1][1],
                                 z1=0,
                                 z2=0,
                                 radius1=0.002,
                                 radius2=0.002,
                                 radius=0.002,
                                 length=0,
                                 order=0,
                                 insertion=None
                                 )

    base_segment_right = g_left.node(id_segment)

    polyline_to_mtg_vid[ids.index(right_seed_id)] = base_segment_right.index()

    # We initialize an empty dictionary that will be used to register the vid of the mother elements:
    index_pointer_in_mtg = {}
    index_pointer_in_mtg[0] = {}
    index_pointer_in_mtg[0][0] = base_segment_right.index()
    # We initialize the first mother element:
    mother_element = base_segment_right

    print("Opening 2D RSML...")
    
    current_ids = [right_seed_id.replace(" ", "")]
    
    iterating = True

    while iterating:
        tp_child_list = []
        iterating = False
        for current_id in current_ids:
            current_axis = g_right.Axis(polyline_to_mtg_vid[ids.index(current_id)])
            children_axes = [child_id.replace(" ", "") for child_id, parent_id in translator.items() if parent_id.replace(" ", "") == current_id.replace(" ", "")]
            if len(children_axes) > 0:
                iterating = True

            for child_id in children_axes:
                child_index = ids.index(child_id)

                line = polylines[child_index]
                child_insertion = insertions[child_index]

                i_x1, i_y1 = line[0]
                i_x2, i_y2 = line[1]
                i_length=np.sqrt((i_x2 - i_x1)**2 + (i_y2 - i_y1)**2)
                
                insertion_x = i_x1 + child_insertion * (i_x1 - i_x2) / i_length
                insertion_y = i_y1 + child_insertion * (i_y1 - i_y2) / i_length

                distances_to_insertion = {}
                for vid in current_axis:
                    n = g_right.node(vid)
                    distance = np.sqrt(((insertion_x - (n.x1 + n.x2) / 2)**2) + ((insertion_y - (n.y1 + n.y2) / 2)**2))
                    if distance < 0.04:
                        distances_to_insertion[vid] = distance
                
                if len(distances_to_insertion) > 0:
                    closest_vid = min(distances_to_insertion, key=distances_to_insertion.get)

                    mother_element = g_right.node(closest_vid)

                    for i in range(1, len(line)):
                        if i==1:
                            # If this is the first element of the axis, and this is not the collar point, this element is a ramification.
                            edgetype="+"
                            order = mother_element.order + 1
                        else:
                            edgetype="<"
                            order = mother_element.order
                        insertion=child_insertion

                        # We define the label (Apex or Segment):
                        if i == len(line) - 1:
                            label="Apex"
                        else:
                            label="Segment"

                        x1, y1 = line[i-1]
                        z1 = 0
                        x2, y2 = line[i]
                        z2 = 0

                        r1 = functions["diameter"][child_index][i - 1] / 2
                        r2 = functions["diameter"][child_index][i] / 2

                        length=np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

                        new_child = mother_element.add_child(edge_type=edgetype,
                                                    label=label,
                                                    type="Normal_root_after_emergence",
                                                    x1=x1,
                                                    x2=x2,
                                                    y1=y1,
                                                    y2=y2,
                                                    z1=-z1,
                                                    z2=-z2,
                                                    radius1=r1,
                                                    radius2=r2,
                                                    radius=(r1+r2)/2,
                                                    length=length,
                                                    order=order,
                                                    insertion=insertion)
                        
                        if i == 1:
                            polyline_to_mtg_vid[child_index] = new_child.index()

                        mother_element = new_child

                    tp_child_list.append(child_id)

        current_ids = tp_child_list

    
    # Mapping annotations
    print("Mapping annotations")
    vertices = g_right.vertices(scale=g_right.max_scale())
    props = g_right.properties()
    loading = 0
    for annotation in annotations:
        print('.'*(loading%10), flush=True, end='\r')
        x_measure = annotation['x']
        y_measure = annotation['y']
        
        min_distance = 1
        closest_vid = -1
        for vid in vertices:
            n = g_right.node(vid)
            distance = np.sqrt( ((x_measure - (n.x1 + n.x2)/2) ** 2) + ((y_measure - (n.y1 + n.y2)/2) ** 2) )
            if distance < min_distance:
                min_distance = distance
                closest_vid = vid
        
        if annotation["label"] not in props:
            props.setdefault(annotation["label"], {})
        
        if min_distance < 0.01:
            props[annotation["label"]][closest_vid] = annotation["value"]

        loading += 1
        if loading%10 == 0:
            print(" "*10, flush=True, end='\r')

    # for vid in pre_order2(g, root):
    #     n = g.node(vid)
    #     print(n.order)

    return g_right


def read_rsml(name: str, two_dimensionnal=True):
    """Parses the RSML file into:

    Args:
    name(str): file name of the rsml file

    Returns:
    (list, dict, dict):
    (1) a (flat) list of polylines, with one polyline per root
    (2) a dictionary of properties, one per root, adds "parent_poly" holding the index of the parent root in the list of polylines
    (3) a dictionary of functions
    """
    root = ET.parse(name).getroot()

    polylines = []
    annotations = []
    ids = []
    insertions = []
    properties = {}
    functions = {}
    for scene in root.iter():
        for plant in scene.iter():
            for elem in plant.iter():
                (polylines, properties, functions, annotations) = parse_rsml_(elem, polylines, properties, functions, annotations, -1)
                if "ID" in elem.attrib:
                    ids.append(elem.attrib["ID"])
                    if elem[0].tag == 'properties':
                        found_insertion = -1
                        for prop in elem[0].iter():
                            if prop.tag == 'insertion':
                                found_insertion = float(prop.text)
                        insertions.append(found_insertion)

                if "label" in elem.attrib:
                    if elem.attrib["label"] == 'seed_L':
                        for poly in elem.iterfind('geometry'):
                            polyline = []
                            for p in poly[0]:  # 0 is the polyline
                                n = p.attrib
                                if two_dimensionnal:
                                    newnode = [float(n['x']), float(n['y'])]
                                else:
                                    newnode = [float(n['x']), float(n['y']), float(n['z'])]
                                polyline.append(newnode)
                        left_seed = polyline
                    if elem.attrib["label"] == 'seed_R':
                        for poly in elem.iterfind('geometry'):
                            polyline = []
                            for p in poly[0]:  # 0 is the polyline
                                n = p.attrib
                                if two_dimensionnal:
                                    newnode = [float(n['x']), float(n['y'])]
                                else:
                                    newnode = [float(n['x']), float(n['y']), float(n['z'])]
                                polyline.append(newnode)
                        right_seed = polyline

            return polylines, properties, functions, annotations, right_seed, left_seed, ids, insertions


def parse_rsml_(organ: ET, polylines: list, properties: dict, functions: dict, annotations: list, parent: int, two_dimensionnal = True):
    """ Recursivly parses the rsml file, used by read_rsml """
    for poly in organ.iterfind('geometry'):
        polyline = []
        for p in poly[0]:  # 0 is the polyline
            n = p.attrib
            if two_dimensionnal:
                newnode = [float(n['x']), float(n['y'])]
            else:
                newnode = [float(n['x']), float(n['y']), float(n['z'])]
            polyline.append(newnode)
        polylines.append(polyline)
        properties.setdefault("parent-poly", []).append(parent)

    # for prop in organ.iterfind('properties'):
    #     for p in prop:  # i.e legnth, type, etc..
    #         try:
    #             value = float(p.attrib['value'])
    #         except ValueError:
    #             value = p.attrib['value']
    #         properties.setdefault(str(p.tag), []).append(value)


    for funcs in organ.iterfind('functions'):
        for fun in funcs:
            samples = []
            for sample in fun.iterfind('sample'):
                value = float(getattr(sample, 'text'))
                samples.append(value)
            functions.setdefault(str(fun.attrib['name']), []).append(samples)

    for annots in organ.iterfind('annotations'):
        for annotation in annots:
            if "name" in annotation.attrib:
                annotation_dict = {}
                annotation_dict["label"] = annotation.attrib["name"]
                for sample in annotation:
                    if sample.tag == "point":
                        x = float(sample.attrib["x"])
                        y = float(sample.attrib["y"])
                        annotation_dict["x"] = x
                        annotation_dict["y"] = y
                    if sample.tag == "value":
                        annotation_dict["value"] = sample.text
                annotations.append(annotation_dict)

    pi = len(polylines) - 1
    for elem in organ.iterfind('root'):  # and all laterals
        polylines, properties, functions, annotations = parse_rsml_(elem, polylines, properties, functions, annotations, pi)

    return polylines, properties, functions, annotations


def mtg_from_rsml_griffith(file_path: str, length_unit_conversion_factor = 54.0e-6, min_length=4e-3, diameter_filter_threshold: float = 0.5):
    """
    param: min_lengt in m
    """

    polylines, properties, functions = read_rsml(file_path)

    for f, v in functions.items():
        functions[f] = [[value * length_unit_conversion_factor if value else None for value in l ] for l in v]
    
    origin = polylines[0][0]

    polylines = [[[(i[j] - origin[j]) * length_unit_conversion_factor for j in range(len(origin))] for i in k] for k in polylines]

    if len(polylines[0][0]) == 2:
        flat_rsml = True
    elif len(polylines[0][0]) == 3:
        flat_rsml = False
    else:
        raise SyntaxError("Error in RSML file format, wrong number of coordinates")

    # We create an empty MTG:
    g = MTG()

    # We define the first base element as an empty element:
    if flat_rsml:
        x1, y1 = polylines[0][0]
        z1 = 0
    else:
        x1, y1, z1 = polylines[0][0]
    
    if not functions["diameter"][0][0]:
        r1 = 0.002
    else:
        r1 = functions["diameter"][0][0] / 2.

    id_segment = g.add_component(g.root, label='Segment',
                                 type="Base_of_the_root_system",
                                 x1=x1,
                                 x2=x1,
                                 y1=y1,
                                 y2=y1,
                                 z1=-z1,
                                 z2=-z1,
                                 radius1=r1,
                                 radius2=r1,
                                 radius=r1,
                                 length=0,
                                 order=1
                                 )
    base_segment = g.node(id_segment)

    # We initialize an empty dictionary that will be used to register the vid of the mother elements:
    index_pointer_in_mtg = {}
    index_pointer_in_mtg[0] = {}
    index_pointer_in_mtg[0][0] = base_segment.index()
    # We initialize the first mother element:
    mother_element = base_segment

    if flat_rsml:
        print("Opening 2D RSML...")
    else: 
        print("Opening 3D RSML...")
    
    # For each root axis:
    for l, line in enumerate(polylines):
        mean_radius_axis = np.mean([k for k in functions["diameter"][l] if k]) / 2
        
        # If the root axis is not the main one of the root system:
        if l > 0 and len(line) > 0:
            # We initialize the first dictionary within the main dictionary:
            index_pointer_in_mtg[l] = {}

            # We define the mother element of the current lateral axis according to the properties of the RSML file:
            parent_axis_index = properties["parent-poly"][l]
            if "parent-node" in properties.keys():
                parent_node_index = properties["parent-node"][l]
            else:
                insertion_distances = [np.sqrt((x-line[0][0])**2 + (y-line[0][1])**2 + (z-line[0][2])**2) for (x, y, z) in polylines[parent_axis_index]]
                parent_node_index = insertion_distances.index(min(insertion_distances))

            mother_element = g.node(index_pointer_in_mtg[parent_axis_index][parent_node_index])

        # For each root element:
        for i in range(1,len(line)):
            # We define the x,y,z coordinates and the radius of the starting and ending point:
            if flat_rsml:
                x1, y1 = line[i-1]
                z1 = 0
                x2, y2 = line[i]
                z2 = 0
            else:
                x1, y1, z1 = line[i-1]
                x2, y2, z2 = line[i]

            if not functions["diameter"][l][i - 1]:
                r1 = mean_radius_axis
            else:
                r1 = functions["diameter"][l][i - 1] / 2

            if not functions["diameter"][l][i]:
                r2 = mean_radius_axis
            else:
                r2 = functions["diameter"][l][i] / 2

            # Filtering cases where incoherent high or low diameters have been annotated
            if r1 > mean_radius_axis * (1 + diameter_filter_threshold) or r1 < mean_radius_axis * (1 - diameter_filter_threshold):
                r1 = mean_radius_axis
            
            if r2 > mean_radius_axis * (1 + diameter_filter_threshold) or r2 < mean_radius_axis* (1 - diameter_filter_threshold):
                r2 = mean_radius_axis
                
            # The length of the root element is calculated from the x,y,z coordinates:
            length=np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

            # We define the edge type ('<': adding a root element on the same axis, '+': adding a lateral root):
            if i==1 and l > 0:
                # If this is the first element of the axis, and this is not the collar point, this element is a ramification.
                edgetype="+"
                order = mother_element.order + 1
            else:
                edgetype="<"
                order = mother_element.order
            # We define the label (Apex or Segment):
            if i == len(line) - 1:
                label="Apex"
            else:
                label="Segment"

            if mother_element.length < min_length and edgetype == "<":
                mother_element.x2 = x2
                mother_element.y2 = y2
                mother_element.z2 = -z2
                mother_element.radius = (mother_element.radius + r2) / 2
                mother_element.r2 = r2
                mother_element.length = np.sqrt(  (mother_element.x2-mother_element.x1)**2 
                                                + (mother_element.y2-mother_element.y1)**2 
                                                + (mother_element.z2-mother_element.z1)**2)
                if label == "Apex":
                    mother_element.label = "Apex"
                
                index_pointer_in_mtg[l][i]=mother_element.index()
            else:
                # We finally add the new root element to the previously-defined mother element:
                new_child = mother_element.add_child(edge_type=edgetype,
                                                    label=label,
                                                    type="Normal_root_after_emergence",
                                                    x1=x1,
                                                    x2=x2,
                                                    y1=y1,
                                                    y2=y2,
                                                    z1=-z1,
                                                    z2=-z2,
                                                    radius1=r1,
                                                    radius2=r2,
                                                    radius=(r1+r2)/2,
                                                    length=length,
                                                    order=order)
                # We record the vertex ID of the current root element:
                vid = new_child.index()
                # We add the vid to the dictionary:
                index_pointer_in_mtg[l][i]=vid
                # And we now consider current element as the mother element for the next iteration on this axis:
                mother_element = new_child
    
    # Finally, we filter diameters that might have remained too high because whole axis was wrong
    per_order_mean_diameters = {order: np.mean([g.property("radius")[k] for k in g.vertices() if k != 0 and g.property("order")[k] == order]) for order in [1, 2, 3, 4, 5]}

    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    for vid in pre_order2(g, root):
        n = g.node(vid)
        parent = n.parent()
        if parent:
            if n.radius > parent.radius * (1 + diameter_filter_threshold):
                n.radius = per_order_mean_diameters[n.order]

    return g

def read_rsml_griffith(name: str):
    """Parses the RSML file into:

    Args:
    name(str): file name of the rsml file

    Returns:
    (list, dict, dict):
    (1) a (flat) list of polylines, with one polyline per root
    (2) a dictionary of properties, one per root, adds "parent_poly" holding the index of the parent root in the list of polylines
    (3) a dictionary of functions
    """
    root = ET.parse(name).getroot()
    plant = root[1][0]
    polylines = []
    properties = {}
    functions = {}
    for elem in plant.iterfind('root'):
        (polylines, properties, functions) = parse_rsml_(elem, polylines, properties, functions, -1)

    return polylines, properties, functions


def parse_rsml_griffith(organ: ET, polylines: list, properties: dict, functions: dict, parent: int):
    """ Recursivly parses the rsml file, used by read_rsml """
    for poly in organ.iterfind('geometry'):  # only one
        polyline = []
        for p in poly[0]:  # 0 is the polyline
            n = p.attrib
            newnode = [float(n['x']), float(n['y']), float(n['z'])]
            polyline.append(newnode)
        polylines.append(polyline)
        properties.setdefault("parent-poly", []).append(parent)

    for prop in organ.iterfind('properties'):
        for p in prop:  # i.e legnth, type, etc..
            try:
                value = float(p.attrib['value'])
            except ValueError:
                value = p.attrib['value']
            properties.setdefault(str(p.tag), []).append(value)


    for funcs in organ.iterfind('functions'):
        for fun in funcs:
            samples = []
            for sample in fun.iterfind('sample'):
                try:
                    value = float(sample.attrib['value'])
                except ValueError:
                    value = None
                samples.append(value)
            functions.setdefault(str(fun.attrib['name']), []).append(samples)

    pi = len(polylines) - 1
    for elem in organ.iterfind('root'):  # and all laterals
        polylines, properties, functions = parse_rsml_(elem, polylines, properties, functions, pi)

    return polylines, properties, functions