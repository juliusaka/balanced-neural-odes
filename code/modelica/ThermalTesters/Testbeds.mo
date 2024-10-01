within ThermalTesters;
package Testbeds
  model StratifiedHeatFlowModelTester
    StratifiedHeatFlowModel stratifiedHeatFlowModel
               annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
    Modelica.Blocks.Sources.RealExpression realExpression(y=293.15)
                                                                annotation (Placement(transformation(extent={{80,-10},{60,10}})));
    Modelica.Blocks.Sources.Step step(height=100, offset=273.15)
                                                             annotation (Placement(transformation(extent={{-100,20},{-80,40}})));
    Modelica.Blocks.Sources.Sine sine annotation (Placement(transformation(extent={{-100,-20},{-80,0}})));
  equation
    connect(step.y, stratifiedHeatFlowModel.temperature_K_a) annotation (Line(points={{-79,30},{-16,30},{-16,-3.33333},{-7.14286,-3.33333}},
                                                                                                                          color={0,0,127}));
    connect(stratifiedHeatFlowModel.temperature_K_b, realExpression.y) annotation (Line(points={{7.14286,-3.33333},{34,-3.33333},{34,0},{59,0}},
                                                                                                                color={0,0,127}));
    annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)),
      experiment(Interval=0.002, __Dymola_Algorithm="Dassl"));
  end StratifiedHeatFlowModelTester;

  model ThreeStratifiedHeatFlowModelTester
    ThreeStratifiedHeatFlowModels
                            threeStratifiedHeatFlowModels(
      nSeg=16) annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
    Modelica.Blocks.Sources.RealExpression realExpression(y=20) annotation (Placement(transformation(extent={{80,-10},{60,10}})));
    Modelica.Blocks.Sources.Step step(height=100, offset=20) annotation (Placement(transformation(extent={{-100,20},{-80,40}})));
    Modelica.Blocks.Sources.Sine sine annotation (Placement(transformation(extent={{-100,-20},{-80,0}})));
  equation
    connect(step.y, threeStratifiedHeatFlowModels.temperature_K_a) annotation (Line(points={{-79,30},{-16,30},{-16,0},{-5.55556,0}},
                                                                                                                                color={0,0,127}));
    connect(threeStratifiedHeatFlowModels.temperature_K_b, realExpression.y) annotation (Line(points={{5.55556,0},{59,0}},
                                                                                                                      color={0,0,127}));
    annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)));
  end ThreeStratifiedHeatFlowModelTester;

  model StratifiedHeatFlowModelTester_ROM
    Modelica.Blocks.Sources.RealExpression realExpression(y=293.15)
                                                                annotation (Placement(transformation(extent={{92,70},{72,90}})));
    Modelica.Blocks.Sources.Step step(height=100, offset=273.15)
                                                             annotation (Placement(transformation(extent={{-100,80},{-80,100}})));
    Modelica.Blocks.Sources.Sine sine(
      amplitude=20,
      f=0.01,
      offset=293.15)                  annotation (Placement(transformation(extent={{-100,40},{-80,60}})));
    StratifiedHeatFlowModel_ROM stratifiedHeatFlowModel_ROM(p=16, n_r=8)
                                                                        annotation (Placement(transformation(extent={{-14,10},{14,16}})));
    StratifiedHeatFlowModel_forLinearization stratifiedHeatFlowModel_Linearization annotation (Placement(transformation(extent={{-14,74},{14,80}})));
  equation
    connect(stratifiedHeatFlowModel_Linearization.temperature_K_b, realExpression.y) annotation (Line(points={{10,76},{44,76},{44,80},{71,80},{71,80}}, color={0,0,127}));
    connect(realExpression.y, stratifiedHeatFlowModel_ROM.temperature_K_b) annotation (Line(points={{71,80},{44,80},{44,76},{18,76},{18,12},{10,12}}, color={0,0,127}));
    connect(step.y, stratifiedHeatFlowModel_Linearization.temperature_K_a) annotation (Line(points={{-79,90},{-18,90},{-18,70},{-10,70},{-10,76}}, color={0,0,127}));
    connect(step.y, stratifiedHeatFlowModel_ROM.temperature_K_a) annotation (Line(points={{-79,90},{-18,90},{-18,6},{-10,6},{-10,12}}, color={0,0,127}));
    annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)),
      experiment(Interval=0.002, __Dymola_Algorithm="Dassl"));
  end StratifiedHeatFlowModelTester_ROM;
end Testbeds;
