within ClaRaTester;
model Testbed
  ClaRaTester.SteamCycle_01 steamCycle_01_1 annotation (Placement(transformation(extent={{-8,-10},{12,10}})));
  Modelica.Blocks.Sources.TimeTable PTarget(table=[0,0.8; 500,0.8; 510,1.0; 1400,0.7; 1410,1; 2300,1; 2310,0.7; 3200,0.7; 3210,1; 5000,1]) annotation (Placement(transformation(extent={{-68,-10},{-48,10}})));
equation
  connect(PTarget.y, steamCycle_01_1.u) annotation (Line(points={{-47,0},{-30.8,0}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=5000,
      Tolerance=1e-05,
      __Dymola_Algorithm="Dassl"));
end Testbed;
