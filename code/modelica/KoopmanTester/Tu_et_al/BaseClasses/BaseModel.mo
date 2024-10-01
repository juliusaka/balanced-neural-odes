within KoopmanTester.Tu_et_al.BaseClasses;
partial model BaseModel
        Real x1;
        Real x2;
        parameter Real lambda = -1.0;
        parameter Real mu = -0.5;
        parameter Real x1_initial = 10;
        parameter Real x2_initial = 20;

  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
        coordinateSystem(preserveAspectRatio=false)));
end BaseModel;
