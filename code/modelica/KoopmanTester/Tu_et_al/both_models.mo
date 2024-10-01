within KoopmanTester.Tu_et_al;
model both_models "includes both models (nonlinear and Koopman Version)"
  NonlinearModel nonlinearModel annotation (Placement(transformation(extent={{-60,0},{-40,20}})));
  KoopmanVersion koopmanVersion annotation (Placement(transformation(extent={{20,2},{40,22}})));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)));
end both_models;
