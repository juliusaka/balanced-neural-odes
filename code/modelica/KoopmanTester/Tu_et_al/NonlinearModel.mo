within KoopmanTester.Tu_et_al;
model NonlinearModel "From Brunton et. al. 2021 Koopman Review"
  extends BaseClasses.BaseModel;
equation
  der(x1) = mu * x1;
  der(x2) = lambda * ( x2 - x1^2);

  annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={
          Rectangle(
          extent={{-60,60},{60,-60}},
          lineColor={28,108,200},
          fillColor={255,0,0},
          fillPattern=FillPattern.Solid), Text(
          extent={{-48,32},{48,-36}},
          textColor={28,108,200},
          textString="Tu et. al.")}),                            Diagram(
        coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=10,
      __Dymola_NumberOfIntervals=100,
      Tolerance=1e-05,
      __Dymola_Algorithm="Dassl"));
end NonlinearModel;
