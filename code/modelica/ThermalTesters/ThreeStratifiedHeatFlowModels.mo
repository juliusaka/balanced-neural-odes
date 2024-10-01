within ThermalTesters;
model ThreeStratifiedHeatFlowModels
  extends Modelica.Icons.Example;
  BaseClasses.StratifiedHeatFlow stratifiedHeatFlow1(
    nSeg=nSeg,
    C=C1,
    R=R1) annotation (Placement(transformation(extent={{-40,-10},{-20,10}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature prescribedTemperature_a annotation (Placement(transformation(extent={{-70,-10},{-50,10}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature prescribedTemperature_b annotation (Placement(transformation(extent={{70,-10},{50,10}})));
  Modelica.Blocks.Interfaces.RealInput temperature_K_a(start=373.15) annotation (Placement(transformation(extent={{-120,-20},{-80,20}})));
  Modelica.Blocks.Interfaces.RealInput temperature_K_b(start=373.15) annotation (Placement(transformation(extent={{120,-20},{80,20}})));
  parameter Integer nSeg=10;
  BaseClasses.StratifiedHeatFlow stratifiedHeatFlow2(
    nSeg=nSeg,
    C=C2,
    R=R2)                                           annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  BaseClasses.StratifiedHeatFlow stratifiedHeatFlow3(
    nSeg=nSeg,
    C=C3,
    R=R3)                                           annotation (Placement(transformation(extent={{20,-10},{40,10}})));
  parameter Modelica.Units.SI.HeatCapacity C1=0.2 "Heat capacity of all elements (= cp*m)";
  parameter Modelica.Units.SI.ThermalResistance R1=0.3 "Constant thermal resistance of one element";
  parameter Modelica.Units.SI.HeatCapacity C2=1.0 "Heat capacity of all elements (= cp*m)";
  parameter Modelica.Units.SI.ThermalResistance R2=0.1 "Constant thermal resistance of one element";
  parameter Modelica.Units.SI.HeatCapacity C3=0.6 "Heat capacity of all elements (= cp*m)";
  parameter Modelica.Units.SI.ThermalResistance R3=0.6 "Constant thermal resistance of one element";
equation
  connect(prescribedTemperature_b.T, temperature_K_b) annotation (Line(points={{72,0},{100,0}}, color={0,0,127}));
  connect(temperature_K_a, prescribedTemperature_a.T) annotation (Line(points={{-100,0},{-72,0}}, color={0,0,127}));
  connect(prescribedTemperature_a.port, stratifiedHeatFlow1.port_a) annotation (Line(points={{-50,0},{-40,0}}, color={191,0,0}));
  connect(stratifiedHeatFlow1.port_b, stratifiedHeatFlow2.port_a) annotation (Line(points={{-19.8,0},{-10,0}}, color={191,0,0}));
  connect(stratifiedHeatFlow2.port_b,stratifiedHeatFlow3. port_a) annotation (Line(points={{10.2,0},{20,0}}, color={191,0,0}));
  connect(stratifiedHeatFlow3.port_b, prescribedTemperature_b.port) annotation (Line(points={{40.2,0},{50,0}}, color={191,0,0}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false, extent={{-180,-60},{180,60}})),
                                                                 Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-180,-60},{180,60}})));
end ThreeStratifiedHeatFlowModels;
