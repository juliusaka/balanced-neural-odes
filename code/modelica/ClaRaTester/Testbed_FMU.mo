within ClaRaTester;
model Testbed_FMU
  ClaRaTester_SteamCycle_fmu claRaTester_SteamCycle_fmu(_u_start=_u_start)
                                                        annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  Modelica.Blocks.Sources.TimeTable PTarget(table=[0,0.8; 500,0.8; 510,1.0; 1400,0.7; 1410,1; 2300,1; 2310,0.7; 3200,0.7; 3210,1; 5000,1])
                                                                                                                                         annotation (Placement(transformation(extent={{-62,-10},{-42,10}})));
  parameter Real _u_start( fixed=false);

initial equation
  _u_start = PTarget.y;
equation
  connect(PTarget.y, claRaTester_SteamCycle_fmu.u) annotation (Line(points={{-41,0},{-10.4,0}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=5000,
      Tolerance=1e-06,
      __Dymola_Algorithm="Dassl"));
end Testbed_FMU;
