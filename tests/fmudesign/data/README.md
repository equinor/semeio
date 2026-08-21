Text test data for `fmudesign`.

`config` contains domain-level YAML specifications for representative inputs:
general settings, sensitivities, defaults, correlations, dependencies, and
auxiliary tables. The shared renderer derives the Excel layout and creates
workbooks in a temporary directory for integration and regression tests.
`design_input_background_extseeds.yaml` owns `seeds.xlsx` and the shared
`doe1.xlsx` table referenced by several configurations in this directory.
