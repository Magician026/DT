# Official published comparison

Source: UniVTAC paper Table I, https://shanluo.github.io/ViTacWorkshops/assets/accepted_paper/pdf/2.pdf

|Task|Official ACT+UniVTAC|Details V2|Difference|
|---|---:|---:|---:|
|Insert HDMI policy42|28%|31%|+3pp|
|Insert HDMI policy1|28%|23%|-5pp|
|Insert Tube policy0|56%|58%|+2pp|
|Pull Out Key policy0|46%|36%|-10pp|

HDMI two-policy mean is27%, not above official28%. Key has100 completed distinct rollouts and0 infrastructure errors; Wilson95%CI27.3–45.8%. This is a descriptive paper comparison, not a paired controlled significance test.

Key alone used depth-only encoder supervision after marker-reference validation failed; Tube and HDMI used depth+marker. Therefore task and pretraining-recipe differences are confounded. Missing shear/displacement supervision is a plausible hypothesis, not a demonstrated cause. No architecture or hyperparameters were changed after seeing outcomes. Priority: diagnose failure modes and marker-reference provenance before another controlled experiment. Single-policy variation remains unresolved for Tube/Key.
