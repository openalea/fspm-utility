import os
import sys
import pickle
import timeit
import xarray as xr
import pandas as pd
from dataclasses import fields

class Logger:
    def __init__(self, model_instance, outputs_dirpath="", 
                 output_variables={}, scenario={"default":1}, time_step_in_hours=1, 
                 logging_period_in_hours=1, 
                 recording_sums=True, recording_raw=True, recording_mtg=True, recording_images=True, echo=True):
        self.g = model_instance.g
        self.models = model_instance.models
        self.outputs_dirpath = outputs_dirpath
        self.output_variables = output_variables
        self.scenario = scenario
        self.summable_output_variables = []
        self.time_step_in_hours = time_step_in_hours
        self.logging_period_in_hours = logging_period_in_hours
        self.recording_sums = recording_sums
        self.recording_raw = recording_raw
        self.recording_mtg = recording_mtg
        self.recording_images = recording_images
        self.echo = echo
        # TODO : add a scenario named folder
        self.root_images_dirpath = os.path.join(self.outputs_dirpath, "root_images")
        self.MTG_files_dirpath = os.path.join(self.outputs_dirpath, "MTG_files")
        self.MTG_properties_summed_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/MTG_properties_summed")
        self.MTG_properties_raw_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/MTG_properties_raw")
        self.create_or_empty_directory(self.outputs_dirpath)
        self.create_or_empty_directory(self.root_images_dirpath)
        self.create_or_empty_directory(self.MTG_files_dirpath)
        self.create_or_empty_directory(self.MTG_properties_summed_dirpath)
        self.create_or_empty_directory(self.MTG_properties_raw_dirpath)

        if self.output_variables == {}:
            for model in self.models:
                self.summable_output_variables += model.extensive_variables
                self.output_variables.update({f.name:f.metadata for f in fields(model) if f.name in model.state_variables})

        self.summed_variables = pd.DataFrame(columns=self.summable_output_variables)
        self.log_xarray =  []

        self.start_time = timeit.default_timer()
        self.previous_step_start_time = self.start_time
        self.simulation_time_in_hours = 0

    def create_or_empty_directory(self, directory=""):
        if not os.path.exists(directory):
        # We create it:
            os.mkdir(directory)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    os.remove(os.path.join(root, file))

    def recording_summed_MTG_properties_to_csv(self):
        step_sum = pd.DataFrame({var:sum(self.g[var].values()) for var in self.summable_output_variables})
        self.summed_variables = self.summed_variables.append(step_sum)

    def recording_raw_MTG_properties_in_xarray(self):
        self.log_xarray += [self.mtg_to_dataset(self.g, variables=self.output_variables, time=self.simulation_time_in_hours)]
        if sys.getsizeof(self.log_xarray) > 2000:
            self.write_to_disk(self.log_xarray)
            # Check save maybe
            self.log_xarray = []

    def mtg_to_dataset(self, variables, 
                       coordinates=dict(vid=dict(unit="adim", value_example=1, description="Root segment identifier index"),
                                        t=dict(unit="h", value_example=1, description="Model time step")),
                       description="Model local root MTG properties over time", 
                       time=0):
        # convert dict to dataframe with index corresponding to coordinates in topology space
        # (not just x, y, z, t thanks to MTG structure)
        props_df = pd.DataFrame.from_dict(self.g.properties())
        props_df["vid"] = props_df.index
        props_df["t"] = [time for k in range(props_df.shape[0])]
        props_df = props_df.set_index(list(coordinates.keys()))

        # Select properties actually used in the current version of the target model
        props_df = props_df[list(variables.keys())]

        # Filter duplicated indexes
        props_df = props_df[~props_df.index.duplicated()]

        # Remove false root segments created just for branching regularity issues (vid 0, 2, 4, etc)
        props_df = props_df[props_df["struct_mass"] > 0]

        # Convert to xarray with given dimensions to spatialize selected properties
        props_ds = props_df.to_xarray()

        # Dataset global attributes
        props_ds.attrs["description"] = description

        # Dataset coordinates' attribute metadata
        for k, v in coordinates.items():
            getattr(props_ds, k).attrs.update(v)

        # Dataset variables' attribute metadata
        for k, v in variables.items():
            getattr(props_ds, k).attrs.update(v)

        return props_ds
    
    def recording_mtg_files(self):
        with open(self.MTG_files_dirpath, "wb") as f:
            pickle.dump(self.g, f)

    def recording_images(self):
        pass
    
    def write_to_disk(self, xarray_list):
        interstitial_dataset = xr.concat(xarray_list, dim="t")
        interstitial_dataset.to_netcdf(os.path.join(self.MTG_properties_raw_dirpath, f't={self.simulation_time_in_hours}.nc'))

    @property
    def elapsed_time(self):
        return round(timeit.default_timer() - self.start_time, 1)
    
    def __call__(self):
        self.current_step_start_time = self.elapsed_time
        self.simulation_time_in_hours += self.time_step_in_hours
        if self.echo:
            print(f"{self.simulation_time_in_hours} hours | step took {
                round(self.current_step_start_time - self.previous_step_start_time, 1)} s | {
                    self.elapsed_time} s of simulation until now")
            
        if self.simulation_time_in_hours % self.logging_period_in_hours == 0:
            if self.recording_sums:
                self.recording_summed_MTG_properties_to_csv()
            if self.recording_raw:
                self.recording_raw_MTG_properties_in_xarray()
            if self.recording_mtg:
                self.recording_mtg_files()
            if self.recording_images:
                self.recording_images()

        self.previous_step_start_time = self.current_step_start_time
    
    def terminate(self):
        # For saved xarray datasets
        if len(self.log_xarray) > 0:
            self.write_to_disk(self.log_xarray)
            del self.log_xarray
        time_step_files = [self.MTG_properties_raw_dirpath + '/' + name for name in os.listdir(self.MTG_properties_raw_dirpath)]
        time_dataset = xr.open_mfdataset(time_step_files)
        time_dataset = time_dataset.assign_coords(coords=self.scenario).expand_dims(dim=dict(zip(list(self.scenario.keys()), [1 for k in self.scenario])))
        time_dataset.to_netcdf(self.MTG_properties_raw_dirpath + '/merged.nc')
        del time_dataset
        for file in os.listdir(self.MTG_properties_raw_dirpath):
            if '.nc' in file and file != "merged.nc":
                os.remove(self.MTG_properties_raw_dirpath + '/' + file)
        
        # eventually also merge images into video


def test_logger():
    return Logger()


