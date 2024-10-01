within ThermalTesters;
model StratifiedHeatFlowModel_forLinearization "Model that can be used for linearization. Needs an additional output in order for Balanced Truncation to work."
  extends ThermalTesters.StratifiedHeatFlowModel;
  Modelica.Blocks.Interfaces.RealOutput[nSeg] T(unit="K") annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={0,50}), iconTransformation(extent={{-10,-10},{10,10}},
        rotation=90,
        origin={0,110})));
equation
  T =  stratifiedHeatFlow.partition[:].heatCapacitor.T;

end StratifiedHeatFlowModel_forLinearization;
