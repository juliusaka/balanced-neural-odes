within KoopmanTester.Tu_et_al;
model KoopmanVersion "From Brunton et. al. 2021 Koopman Review"
        extends BaseClasses.BaseModel;
        // Koopmann Measurement function variables
        Real y[3](start={x1_initial, x2_initial, x1_initial^2});

        parameter Real A[3,3] = [mu, 0, 0;
        0, lambda, -lambda;
        0, 0, 2 * mu];

equation
  der(y) =  A * y;
  x1 = y[1];
  x2 = y[2];

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
      Tolerance=1e-05,
      __Dymola_Algorithm="Dassl"));
end KoopmanVersion;
