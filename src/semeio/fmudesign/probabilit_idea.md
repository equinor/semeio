# Probabilit - A Python library for Monte Carlo Analysis

**Probabilit** is a Python library for Monte Carlo analysis that provides an open-source alternative to commercial Excel add-ins like @Risk, ModelRisk, and Crystal Ball.
Why not instead use existing libraries like `PyMC` or `SALib`? While both are excellent tools, they serve different purposes:

- **PyMC** focuses on Bayesian inference and Markov Chain Monte Carlo (MCMC) methods. It doesn't provide native support for Latin Hypercube Sampling or straightforward specification of correlation between variables.

- **SALib** is designed primarily for sensitivity analysis and its sampling methods assume independent parameters (see [SALib issue #193](https://github.com/SALib/SALib/issues/193)).

**Probabilit** aims to fill this gap by providing:
- Efficient Latin Hypercube Sampling and random sampling
- Intuitive correlation specification (via the Iman-Conover method)
- Recursive parameter definitions, i.e., the mean of a distribution can itself be a distribution.
- A user-friendly API suitable for both interactive notebooks and production code
- See `Core Features` section for more details.

Some of the hard math was already solved in the last version of fmudesign (https://github.com/equinor/semeio/blob/main/CHANGELOG.md#1180).

The library can also be used as the back-end for applications such as this: https://www.probabilit.com/

This could be useful in:

- fmudesign
- ert
- Cost risk projects
- CCS
- DOT
- DWE (Drilling and Well Estimator, tool to estimate time and cost for drilling operations.)
- Wind

### Core Features

- **Latin Hypercube Sampling** by default (and support for standard random sampling).
- **Correlation** via direct Cholesky or Iman–Conover.
- **Near-Correlation** Matrix Adjustments: provides logic similar to Matlab's `nearcorr` if a correlation matrix is invalid.
- **Flexible Compositions**: Combine random variables in general expressions (e.g., `total_cost = material_cost + labor_cost * overhead`).
- **Random Variables as Parameters**: You can use one random variable as the parameter for another (e.g., `exp.Normal(mean=another_variable, std=1)`).
- **Comprehensive Distribution Support**: Including:
  - **Continuous Distributions**:
    - Normal (Gaussian)
    - Log-Normal
    - Uniform
    - Triangular
    - Beta
    - Beta-PERT (Program Evaluation and Review Technique)
    - Gamma
    - Weibull
    - Exponential
    - Student's t
  - **Discrete Distributions**:
    - Poisson
    - Binomial
    - Negative Binomial
    - Discrete Uniform
  - **Data-Driven Distributions**:
    - Kernel Density Estimation (KDE) from data
- **Built-in Plotting**: Common plotting methods (histograms, box plots, correlation plots, etc.).
- **Lazy Evaluation**: Sampling is deferred until results are explicitly needed (e.g., calling `.sample()`, `.hist()`, or `.report()`).
- **Graph Visualization**: Visualize the computational DAG.
- **Report Generation**: Export a summary of the experiment, including variable definitions, correlation matrices, plots, and statistics, in HTML, PDF, or Markdown.
- **Flexible Distribution Definition**:
  - Define distributions using traditional parameters (mean, std, min, max, etc.)
  - Define distributions using percentiles (P10, P50, P90)
  - Automatic parameter conversion between different specification methods
  - Support for both symmetric and skewed distributions via percentiles

### Lazy Evaluation Requirements

1. **Deferred Sampling**
   All sampling operations happen only when results are actually needed (e.g., a call to `.sample()`).
2. **On-Demand Computation**
   If an expression depends on multiple random variables, samples for those variables will only be generated on first access.
3. **Re-Evaluation on Changes**
   If a random variable's parameters or data change, any expressions or final results that depend on it will naturally be re-evaluated when next accessed.
4. **Eager Evaluation Option**
   When the user explicitly requests it, we can generate and store samples at once for all nodes (though no default caching is performed).

### DAG Structure

**Nodes**:
- **Source Nodes**: Base random variables (e.g., Normal, Uniform, user-fitted distributions).
- **Transform Nodes**: Mathematical operations or any custom function combining one or more nodes.
- **Correlation Nodes**: Store correlation requirements (a matrix plus method like Cholesky or Iman–Conover) and ensure the final correlated samples comply with the matrix (possibly adjusting via a nearcorr-like approach).

Each node tracks:
- **Distribution Parameters** (for source nodes)
- **Correlation Requirements** (if relevant)
- **Transform Functions** (for transform nodes)
- **Sampling State** (e.g., sample size, sampling method)
- **Dependency List** (which other nodes it depends on)
- **Lazy Computation Triggers** (logic to generate or re-generate samples on demand)

**Edges**:
- **Data Flow**: Depicts how random variables and transform nodes feed into other transforms or final expressions.
- **Correlation Requirements**: Connect variables that share a correlation specification.
- **Re-Evaluation Triggers**: Ensures that when a node's definition changes, dependent nodes know they must re-generate samples on the next request.

### Evaluation Process

1. **Build the DAG** with all source variables, transforms, and correlation nodes.
2. **Topological Sort** to determine execution order.
3. **Collect Correlation Requirements** and decide on the correlation application strategy (Cholesky vs. Iman–Conover).
4. **Generate Samples When Accessed**: Only when a user calls `.sample()`, `.hist()`, or `.report()`, the library walks the DAG and produces samples for each required node.
5. **No Caching**: Each time results are requested, sampling or transforms are performed anew (or only stored temporarily in the user's workspace).
6. **Propagate Changes**: If a parameter changes, or new data is fitted, the DAG will reflect that, ensuring the next call to `.sample()` or any plot method uses the updated settings.

### Report Generation

A high-level method such as `exp.report()` can generate a default document with all the standard content:
- **Configuration Overview** (sample size, sampling method, correlation strategy, etc.).
- **Variable List** with definitions (names, distribution types, parameters).
- **Correlation Matrices** used in the experiment.
- **Summary Statistics** (mean, standard deviation, percentiles) for each variable or expression.
- **Visualizations** (histograms, boxplots, correlation plots) for quick inspection.
- **Interpretations or Key Findings** (optional narrative about which variables significantly influence results).
- **Output Formats**: HTML, PDF, Markdown.
- **Customization**: Possibly via YAML or similar config files to tailor the final report.


## Example from Cost Risk Analysis

```python
import pytest
import polars as pl
from probabilit import Probabilit

def test_comprehensive_project_analysis():
    """
    Extended test case using all distributions in a project context,
    ensuring all variables contribute to final results.
    """
    exp = Probabilit(n=10_000, sampling_method='latin_hypercube')

    # === Direct Material and Equipment Costs ===
    steel_cost = exp.Normal.from_percentiles(
        p10=800_000,
        p90=1_200_000,
        name="Steel_Cost"
    )

    maintenance_cost = exp.LogNormal.from_percentiles(
        p10=50_000,
        p90=150_000,
        name="Maintenance_Cost"
    )

    fuel_cost = exp.Uniform(
        min_val=1000,
        max_val=1500,
        name="Daily_Fuel_Cost"
    )

    # Define correlation matrix for cost variables
    cost_corr = pl.DataFrame({
        "variable": ["Steel_Cost", "Maintenance_Cost", "Daily_Fuel_Cost"],
        "Steel_Cost": [1.0, 0.6, 0.4],
        "Maintenance_Cost": [0.6, 1.0, 0.3],
        "Daily_Fuel_Cost": [0.4, 0.3, 1.0]
    })

    # Apply correlations using Iman-Conover method
    steel_cost, maintenance_cost, fuel_cost = exp.correlate(
        correlation_matrix=cost_corr,
        variables=[steel_cost, maintenance_cost, fuel_cost],
        method="iman-conover"
    )

    # Define correlation with uncertainty
    # Instead of using a fixed correlation coefficient, we specify that the correlation
    # itself follows a probability distribution. This is useful when:
    # - Historical data for correlation estimation is limited
    # - Correlations might change over time
    # - We want to analyze sensitivity to correlation assumptions
    # exp.correlate(
    #     variables=[steel_cost, maintenance_cost],
    #     correlation=exp.UncertainCorrelation(
    #         mean=0.6,
    #         std=0.1,
    #         distribution="truncated_normal"
    #     )
    # )

    market_scenario = exp.Trigen(
        pessimistic=0.85,
        most_likely=1.0, 
        optimistic=1.15,
        prob_pessimistic=0.10,
        prob_optimistic=0.90,
        name="Market_Scenario_Multiplier"
    )

    equipment_lifetime = exp.Weibull(
        shape=2.5,
        scale=5000,
        name="Equipment_Lifetime"
    )

    # Equipment replacement cost based on lifetime
    equipment_replacement_cost = 100_000 * (10000 / equipment_lifetime)

    # === Labor and Productivity ===
    productivity = exp.Triangular.from_percentiles(
        p10=85,
        p50=100,
        p90=110,
        name="Productivity_Factor"
    )

    worker_absences = exp.DiscreteUniform(
        min_val=0,
        max_val=5,
        name="Daily_Worker_Absences"
    )

    # Correlate productivity and absences (negative correlation)
    prod_corr = pl.DataFrame({
        "variable": ["Productivity_Factor", "Daily_Worker_Absences"],
        "Productivity_Factor": [1.0, -0.4],
        "Daily_Worker_Absences": [-0.4, 1.0]
    })

    productivity, worker_absences = exp.correlate(
        correlation_matrix=prod_corr,
        variables=[productivity, worker_absences]
    )

    # === Quality and Inspection Costs ===
    quality_score = exp.Beta(
        alpha=5,
        beta=2,
        name="Quality_Score"
    )

    quality_issues = exp.Poisson(
        lambda_param=3.5,
        name="Weekly_Quality_Issues"
    )

    successful_inspections = exp.Binomial(
        n=10,
        p=0.8,
        name="Successful_Inspections"
    )

    certification_attempts = exp.NegativeBinomial(
        r=3,
        p=0.6,
        name="Certification_Attempts"
    )

    # Quality-related costs
    rework_cost = quality_issues * 5000 * (1 - quality_score)
    inspection_cost = (10 - successful_inspections) * 2000
    certification_cost = certification_attempts * 10000

    # === Schedule and Timing ===
    schedule_duration = exp.BetaPERT(
        min_val=8,
        most_likely=10,
        max_val=14,
        name="Schedule_Duration"
    )

    repair_time = exp.Gamma(
        shape=2,
        scale=1.5,
        name="Repair_Time"
    )

    time_between_failures = exp.Exponential(
        rate=0.1,
        name="Time_Between_Failures"
    )

    # Correlate quality and schedule variables
    quality_schedule_corr = pl.DataFrame({
        "variable": ["Quality_Score", "Schedule_Duration", "Repair_Time"],
        "Quality_Score": [1.0, -0.5, -0.3],
        "Schedule_Duration": [-0.5, 1.0, 0.4],
        "Repair_Time": [-0.3, 0.4, 1.0]
    })

    quality_score, schedule_duration, repair_time = exp.correlate(
        correlation_matrix=quality_schedule_corr,
        variables=[quality_score, schedule_duration, repair_time]
    )

    # Schedule impact costs
    downtime_cost = (repair_time / time_between_failures) * schedule_duration * 5000
    schedule_delay_cost = schedule_duration * fuel_cost

    # === Market and Price Variations ===
    price_variation = exp.StudentT(
        df=5,
        name="Price_Variation"
    )

    # Apply market variations to base costs
    market_adjusted_cost = (1 + price_variation * 0.1)

    # === Overhead and Risk Factors ===
    historical_overhead = [0.12, 0.15, 0.14, 0.13, 0.16, 0.15, 0.14]
    overhead_rate = exp.KDE.from_data(
        data=historical_overhead,
        name="Overhead_Rate"
    )

    # === Final Cost Calculation ===
    # Direct costs
    direct_cost = (
        steel_cost * market_scenario + # Market affects material costs
        maintenance_cost +
        equipment_replacement_cost +
        productivity * 1_000_000 * (1 + worker_absences * 0.01)
    ) * market_adjusted_cost

    # Indirect costs
    indirect_cost = (
        rework_cost +
        inspection_cost +
        certification_cost +
        downtime_cost +
        schedule_delay_cost
    )

    # Total project cost
    total_cost = (direct_cost + indirect_cost) * (1 + overhead_rate)

    # === Validation and Reporting ===
    results = total_cost.sample()

    assert len(results) == 10_000
    assert all(cost > 0 for cost in results)

    p10_value = total_cost.percentile(10)
    p50_value = total_cost.percentile(50)
    p90_value = total_cost.percentile(90)
    assert p10_value < p50_value < p90_value

    # Generate outputs
    total_cost.hist(title="Total Project Cost Distribution")
    exp.tornado_plot(total_cost, title="Sensitivity Analysis")

    report = exp.report(
        output_format="html",
        title="Comprehensive Project Risk Analysis",
        description="""
        Complete risk analysis with all variables contributing to final cost:
        - Direct costs (materials, equipment, labor)
        - Quality and inspection impacts
        - Schedule effects
        - Market variations
        - Risk adjustments
        """
    )

    assert report is not None
    exp.save("comprehensive_risk_model.yml")
```

## Example from ert

**TODO**

## Example from DOT

**TODO**

## Example from CCS

**TODO**

## Example from WIND

**TODO**

## Example from DWE

**TODO**

## Example of yml

# comprehensive_risk_model.yml

```yml
metadata:
  name: "Comprehensive Project Risk Analysis"
  description: "Complete risk analysis including direct costs, quality impacts, schedule effects, market variations, and risk adjustments"
  created_at: "2024-03-20T10:00:00Z"
  version: "1.0"
  sample_size: 10000
  sampling_method: "latin_hypercube"

variables:
  # Direct Material and Equipment
  Steel_Cost:
    type: "Normal"
    specification: "percentiles"
    parameters:
      p10: 800000
      p90: 1200000

  Maintenance_Cost:
    type: "LogNormal"
    specification: "percentiles"
    parameters:
      p10: 50000
      p90: 150000

  Daily_Fuel_Cost:
    type: "Uniform"
    parameters:
      min_val: 1000
      max_val: 1500

  Equipment_Lifetime:
    type: "Weibull"
    parameters:
      shape: 2.5
      scale: 5000

  # Labor and Productivity
  Productivity_Factor:
    type: "Triangular"
    specification: "percentiles"
    parameters:
      p10: 85
      p50: 100
      p90: 110

  Daily_Worker_Absences:
    type: "DiscreteUniform"
    parameters:
      min_val: 0
      max_val: 5

  # Quality and Inspection
  Quality_Score:
    type: "Beta"
    parameters:
      alpha: 5
      beta: 2

  Weekly_Quality_Issues:
    type: "Poisson"
    parameters:
      lambda_param: 3.5

  Successful_Inspections:
    type: "Binomial"
    parameters:
      n: 10
      p: 0.8

  Certification_Attempts:
    type: "NegativeBinomial"
    parameters:
      r: 3
      p: 0.6

  # Schedule and Timing
  Schedule_Duration:
    type: "BetaPERT"
    parameters:
      min_val: 8
      most_likely: 10
      max_val: 14

  Repair_Time:
    type: "Gamma"
    parameters:
      shape: 2
      scale: 1.5

  Time_Between_Failures:
    type: "Exponential"
    parameters:
      rate: 0.1

  Price_Variation:
    type: "StudentT"
    parameters:
      df: 5

  Overhead_Rate:
    type: "KDE"
    data: [0.12, 0.15, 0.14, 0.13, 0.16, 0.15, 0.14]

correlations:
  cost_variables:
    method: "iman-conover"
    variables:
      - Steel_Cost
      - Maintenance_Cost
    pairs:
      # Correlation coefficients can themselves be uncertain
      # Here we specify that the correlation between Steel_Cost and Maintenance_Cost
      # follows a truncated normal distribution with mean 0.6 and std 0.1
      # This allows for uncertainty in our correlation estimates to be included in the analysis
      - [Steel_Cost, Maintenance_Cost, {
          mean: 0.6,
          std: 0.1,  # or min/max, or p10/p90
          distribution: "truncated_normal"
        }]

  productivity_variables:
    method: "iman-conover"
    variables:
      - Productivity_Factor
      - Daily_Worker_Absences
    pairs:
      - [Productivity_Factor, Daily_Worker_Absences, -0.4]

  quality_schedule_variables:
    method: "iman-conover"
    variables:
      - Quality_Score
      - Schedule_Duration
      - Repair_Time
    pairs:
      - [Quality_Score, Schedule_Duration, -0.5]
      - [Quality_Score, Repair_Time, -0.3]
      - [Schedule_Duration, Repair_Time, 0.4]

expressions:
  equipment_replacement_cost:
    formula: "100000 * (10000 / Equipment_Lifetime)"

  rework_cost:
    formula: "Weekly_Quality_Issues * 5000 * (1 - Quality_Score)"

  inspection_cost:
    formula: "(10 - Successful_Inspections) * 2000"

  certification_cost:
    formula: "Certification_Attempts * 10000"

  downtime_cost:
    formula: "(Repair_Time / Time_Between_Failures) * Schedule_Duration * 5000"

  schedule_delay_cost:
    formula: "Schedule_Duration * Daily_Fuel_Cost"

  market_adjusted_cost:
    formula: "(1 + Price_Variation * 0.1)"

  direct_cost:
    formula: "(Steel_Cost + Maintenance_Cost + equipment_replacement_cost + Productivity_Factor * 1000000 * (1 + Daily_Worker_Absences * 0.01)) * market_adjusted_cost"

  indirect_cost:
    formula: "rework_cost + inspection_cost + certification_cost + downtime_cost + schedule_delay_cost"

  total_cost:
    formula: "(direct_cost + indirect_cost) * (1 + Overhead_Rate)"

outputs:
  plots:
    - type: "histogram"
      variable: "total_cost"
      title: "Total Project Cost Distribution"
    - type: "tornado"
      variable: "total_cost"
      title: "Sensitivity Analysis"

  report:
    format: "html"
    title: "Comprehensive Project Risk Analysis"
    description: |
      Complete risk analysis with all variables contributing to final cost:
      - Direct costs (materials, equipment, labor)
      - Quality and inspection impacts
      - Schedule effects
      - Market variations
      - Risk adjustments
```
