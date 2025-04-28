
# Fuzzy Controller Project

## General Description
This project simulates the behavior of a control system based on fuzzy logic, applied to a simplified industrial plant and a variant with a complex plant model.  
The main goal is to regulate pressure (`P`) through the control action of heat (`ΔH`), aiming to reach a target pressure (`PO`) in a stable manner.

The project analyzes different configurations:
- Simplified plant (without heat dynamics).
- Complex plant (with heat dynamics dependent on historical behavior).
- Modifications of the fuzzy rule base and membership functions.

## Code Structure
- **P1FSIM.py**     - Original simplified machine (full rule base and membership sets).
- **P1FDIN.py**     - Complex machine (dynamic model with coefficient `c`).
- **P1FSIMcam.py**  - Simplified machine with reduced rule base.
- **P1FSIMcam1.py** - Simplified machine with reduced membership sets and adjusted rule base.

## Required Libraries
Before running any script, make sure you have the following Python libraries installed:
- `numpy`
- `matplotlib`

You can install them using the following command:

```bash
pip install numpy matplotlib
```

## Global Parameters
Inside each file, you can modify the following parameters:
- `P_0` : Initial plant pressure.
- `PO`  : Target pressure.
- `K`   : Plant gain.
- `ITE` : Number of iterations.
- `c`   : Stability coefficient (only in the complex machine).

Additionally, you can select the defuzzification method by changing the value of:

```python
seleccionar_metodo = "centroide"  # It can also be "mean" or "height"
```

Available methods:
- `"centroide"` (Center of gravity)
- `"mean"` (Average of maxima)
- `"altura"` (Maximum value)

## Purpose of Each Configuration
- **P1FSIM.py**     - Evaluate basic fuzzy control with a full rule base and complete membership sets.
- **P1FDIN.py**     - Analyze the dynamic response considering memory effects (factor `c`).
- **P1FSIMcam.py**  - Study the impact of removing critical rules from the control logic.
- **P1FSIMcam1.py** - Optimize the system by reducing membership sets and adjusting rules to improve response speed.

## Final Notes
- Each script automatically generates plots for pressure evolution, control action ΔH, rule activation maps, and trajectory tracking.
- You can dynamically modify the defuzzification method and initial conditions to explore different scenarios.
- The code is modularly designed for easy adjustment or future extensions.
