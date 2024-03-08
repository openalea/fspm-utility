import pandas as pd


class MakeScenarios:

    def from_excel(file_path, which=[]):
        instructions = pd.read_excel(file_path)
        if len(which) > 0:
            scenario_names = which
        else:
            scenario_names = [scenario_name for scenario_name in instructions.columns if scenario_name not in ("Parameter", "Explanation", "Type/ Unit", "Reference_value")]
        scenarios = {name:dict(zip(instructions["Parameter"], instructions[name].replace({'True':True, 'False':False}))) 
                     for name in scenario_names}
        return scenarios