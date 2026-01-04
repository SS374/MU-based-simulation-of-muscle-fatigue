# MVIC Model Simulator

A comprehensive motor unit recruitment and force generation simulator with both Command Line Interface (CLI) and Graphical User Interface (GUI) applications. The simulator implements the Potvin & Fuglevand's 2017 motor unit-based model of muscle fatigue, originally programmed with MATLAB.

## Features

### Core Simulation
- Simulates motor unit recruitment and force generation
- Configurable number of motor units
- Adjustable target force as a fraction of Maximum Voluntary Isometric Contraction (MVIC)
- Customizable task duration
- Detailed force-time and MU recruitment visualization

### CSV Export
- Save simulation results to CSV for further analysis
- Export includes force-time data and motor unit recruitment patterns
- Easy integration with data analysis tools like Excel, R, or Python pandas

### Animation
- Real-time animation of motor unit recruitment (optional)
- Interactive plots with zoom and pan capabilities
- Toggle legends and annotations for better data interpretation

## Installation

### Prerequisites
- Python 3.7 or higher
- Required Python packages:
  ```
  pip install numpy matplotlib PyQt5
  ```

## Applications

### 1. GUI (Recommended)
The `mvic_gui.py` provides a user-friendly interface for interactive simulation and visualization.

#### Features
- Sliders for parameter adjustment
- Real-time command preview
- Built-in console output
- Process status indicators

#### Launching the GUI
```bash
python mvic_gui.py
```

## Output Files
- CSV files are saved in the working directory with timestamps
- Plot windows are interactive and can be saved using the GUI controls

### 1. Command Line Interface (CLI)
The `mvic_cli.py` provides a powerful command-line interface for batch processing and automation.

#### Usage
```bash
python mvic_cli.py [options]
```

#### Options
- `--nu INT`         Number of motor units (default: 200)
- `--fthscale FLOAT` Fraction of MVIC (0.50 = 50% MVC) (default: 0.80)
- `--fthtime FLOAT`  Task duration in seconds (default: 15.0)
- `--save-csv`       Save results to CSV file
- `--animate`        Enable animation of motor unit recruitment

#### Example
```bash
python mvic_cli.py --nu 150 --fthscale 0.75 --fthtime 20 --save-csv --animate
```

## References
This project implements the motor unit-based model of muscle fatigue originally developed by:

Potvin, J. R., & Fuglevand, A. J. (2017). A motor unit-based model of muscle fatigue. *PLoS Computational Biology*, 13(6), e1005581. [https://doi.org/10.1371/journal.pcbi.1005581](https://doi.org/10.1371/journal.pcbi.1005581)

## License
This project is open source and available under the MIT License.
