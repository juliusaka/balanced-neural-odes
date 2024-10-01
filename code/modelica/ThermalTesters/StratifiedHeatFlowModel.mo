within ThermalTesters;
model StratifiedHeatFlowModel
  extends Modelica.Icons.Example;
  BaseClasses.StratifiedHeatFlow stratifiedHeatFlow(
    nSeg=nSeg,
    C=C,
    R=R)                                            annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature prescribedTemperature_a annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature prescribedTemperature_b annotation (Placement(transformation(extent={{60,-10},{40,10}})));
  Modelica.Blocks.Interfaces.RealInput temperature_K_a(start=373.15) annotation (Placement(transformation(extent={{-120,-20},{-80,20}})));
  Modelica.Blocks.Interfaces.RealInput temperature_K_b(start=373.15) annotation (Placement(transformation(extent={{120,-20},{80,20}})));
  parameter Modelica.Units.SI.HeatCapacity C=1.8 "Heat capacity of all elements (= cp*m)";
  parameter Modelica.Units.SI.ThermalResistance R=1 "Constant thermal resistance of one element";
  parameter Integer nSeg=16;
equation
  connect(prescribedTemperature_a.port, stratifiedHeatFlow.port_a) annotation (Line(points={{-40,0},{-10,0}}, color={191,0,0}));
  connect(stratifiedHeatFlow.port_b, prescribedTemperature_b.port) annotation (Line(points={{10.2,0},{40,0}}, color={191,0,0}));
  connect(prescribedTemperature_b.T, temperature_K_b) annotation (Line(points={{62,0},{100,0}}, color={0,0,127}));
  connect(temperature_K_a, prescribedTemperature_a.T) annotation (Line(points={{-100,0},{-62,0}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false, extent={{-140,-20},{140,40}})),
                                                                 Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-140,-20},{140,40}})));
end StratifiedHeatFlowModel;
