#import sdf
import numpy as np

def read_dymola_data(result_file, output_names, start_time, stop_time, step_size):
  '''
  SDF is a package provided by Dymola on pip.
  https://pypi.org/project/SDF/
  :param result_file: path to .mat-result-file
  :type: string
  :param: variable_path_list: list of variable names to read out
  :type: list of strings
  '''
  # sdfData = sdf.load(result_file)
  # time = sdfData["Time"].data
  # results = []
  # for variable in output_names:
  #   splitPath=variable.split(".")
  #   dataGroup=sdfData[splitPath[0]]
  #   if len(splitPath)>1:
  #     for i in splitPath[1:]:
  #       dataGroup=dataGroup[i]
  #   data = dataGroup.data
  #   result = np.interp(np.arange(start_time, stop_time, step_size), time, data)
  #   results.append(result)
  # return np.array(results)
  raise NotImplementedError("read_dymola_data does not work in python 3.10. If you need this, please downgrade")