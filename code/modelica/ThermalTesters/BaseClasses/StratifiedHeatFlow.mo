within ThermalTesters.BaseClasses;
model StratifiedHeatFlow
  parameter Integer nSeg=3;
  parameter Modelica.Units.SI.HeatCapacity C "Heat capacity of all elements (= cp*m)";
  parameter Modelica.Units.SI.ThermalResistance R "Constant thermal resistance of one element";

  R2C1[nSeg] partition(
    each C=C/nSeg,
    each R=R/nSeg);
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a port_a annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_b port_b annotation (Placement(transformation(extent={{92,-10},{112,10}})));
equation
  connect(partition[1].port_a, port_a);
  connect(partition[nSeg].port_b, port_b);
  for i in 1:nSeg-1 loop
    connect(partition[i].port_b, partition[i+1].port_a);
  end for;
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)));
end StratifiedHeatFlow;
