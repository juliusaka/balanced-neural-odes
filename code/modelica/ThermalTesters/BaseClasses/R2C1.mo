within ThermalTesters.BaseClasses;
model R2C1
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor thermalResistor1(R=R/2) annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C=C, T(fixed=true)) annotation (Placement(transformation(extent={{-10,20},{10,40}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor thermalResistor(R=R/2) annotation (Placement(transformation(extent={{40,-10},{60,10}})));
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a port_a annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_b port_b annotation (Placement(transformation(extent={{92,-10},{112,10}})));
  parameter Modelica.Units.SI.HeatCapacity C "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.ThermalResistance R "Constant thermal resistance of material";
equation
  connect(thermalResistor1.port_b, heatCapacitor.port) annotation (Line(points={{-40,0},{0,0},{0,20}}, color={191,0,0}));
  connect(thermalResistor.port_a, heatCapacitor.port) annotation (Line(points={{40,0},{0,0},{0,20}}, color={191,0,0}));
  connect(thermalResistor.port_b, port_b) annotation (Line(points={{60,0},{102,0}}, color={191,0,0}));
  connect(thermalResistor1.port_a, port_a) annotation (Line(points={{-60,0},{-100,0}}, color={191,0,0}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={
        Rectangle(
          extent={{-80,10},{-40,-12}},
          lineColor={28,108,200},
          fillColor={255,213,170},
          fillPattern=FillPattern.Solid),
        Line(points={{20,-6}}, color={28,108,200}),
        Rectangle(
          extent={{40,10},{80,-12}},
          lineColor={28,108,200},
          fillColor={255,213,170},
          fillPattern=FillPattern.Solid),
        Ellipse(
          extent={{-22,50},{22,20}},
          lineColor={28,108,200},
          fillColor={127,0,0},
          fillPattern=FillPattern.Solid),
        Line(
          points={{-40,0},{40,0}},
          color={0,0,0},
          thickness=0.5),
        Line(
          points={{-100,0},{-80,0}},
          color={0,0,0},
          thickness=0.5),
        Line(
          points={{80,0},{100,0}},
          color={0,0,0},
          thickness=0.5),
        Line(
          points={{0,0},{0,20}},
          color={0,0,0},
          thickness=0.5)}),                                      Diagram(coordinateSystem(preserveAspectRatio=false)));
end R2C1;
