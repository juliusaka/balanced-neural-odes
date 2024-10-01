# when exporting fmu

Make the following sure:

- you export an FMU with input connectors, which we can set from the outside
- the input has a defined start-value (start=...) in case inlcude_controls = false
- define/propagate parameters on top level of model. This is a recommendation, such that print_fmu_variable_names.py can print all variables you wish to tune. Otherwise, you can manually include not top level parameters.