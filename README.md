
# **METAMORPHA Point Cloud Dataset and Preprocessing**

## **Overview**
This repository is part of the Horizon Europe Project **METAMORPHA**. It contains the preprocessing script used to process a dataset of 609 point clouds generated via **White Light Interferometry (WLI)**. The dataset also includes a parameter sheet (`data.xlsx`) that summarizes essential metrics derived from the point clouds for machine learning applications.

## **Contents**
- `preprocessing.py`: Python script used to preprocess point clouds and generate the parameter sheet.
- `./data/data.xlsx`: A structured Excel file with processed data metrics.
- `process_point_cloud.ipynb`: A Jupyter notebook explaining step-by-step how preprocessing is applied to a single point cloud file.

## **About the Dataset**
The point cloud dataset contains:
- **Format**: `.xyz` files
- **Number of Files**: 609
- **Description**: Each file represents a surface profile, with metrics derived from roughness calculations and step height analysis.

Due to the size of the dataset (~10 GB), it is not hosted directly on GitHub. If you would like access to the point cloud files, please contact me via email at [micasan4@teleco.upv.es].

## **Preprocessing and Metrics**
The preprocessing script performs the following steps for each `.xyz` file:
1. **Correction of NaN Values**: Replaces missing data.
2. **Tilt Correction**: Aligns surfaces to eliminate measurement tilt.
3. **Filtering (λ < 10 µm)**: Smooths the surface for micro-roughness analysis.
4. **Metric Calculations**:
   - **Sa**: Arithmetic mean roughness.
   - **Sz**: Maximum peak-to-valley height.
   - **Sa λ<10 µm**: Filtered roughness for micro-scale.
   - **Average Step Height**: Difference between reference and processed regions.

The processed metrics are saved in `data.xlsx` for easy use in machine learning pipelines or further analysis.

## **Usage**
To run the preprocessing script:
1. Place the `.xyz` files in the `path/to/point/clouds` directory.
2. Define the output path for `data.xlsx` in the script.
3. Execute the script:
   ```bash
   python preprocessing.py
   ```

## **Generated Output**
The preprocessing script generates an Excel file (`data.xlsx`) with the following columns:
| **Column Name**      | **Description**                                   |
|-----------------------|---------------------------------------------------|
| **Name**             | Name of the `.xyz` file.                         |
| **Sa**               | Roughness (arithmetic mean).                     |
| **Sz**               | Peak-to-valley height.                           |
| **Sa λ<10 µm**       | Filtered roughness for micro-scale.              |
| **Average Step Height** | Step height between reference and processed regions. |

## **Notebook Explanation**
The repository also includes a Jupyter notebook (`notebook_example.ipynb`) that demonstrates, step-by-step, how preprocessing is applied to a single `.xyz` file. This is useful for understanding the methodology in detail.

## **Acknowledgments**
The METAMORPHA project has received funding from Horizon Europe, the European Union’s Framework Programme for Research and Innovation, under grant agreement 101057457.
Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union. The European Union cannot be held responsible for them
