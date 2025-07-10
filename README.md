# ImitateCholec  
**A Multimodal Dataset for Long-Horizon Imitation Learning in Robotic Cholecystectomy**

ImitateCholec is a publicly available dataset designed to advance autonomous robotic systems during the critical clipping and cutting phase of laparoscopic cholecystectomy. It comprises synchronized multi-view video and kinematic recordings from 34 ex vivo porcine procedures, totaling over 18,000 demonstration episodes across 17 fine-grained surgical tasks, including both optimal executions and recovery maneuvers.

![ImitateCholec Overview](images/overview_figure.png)



---

## 🌟 Dataset highlights

- **34** ex vivo porcine cholecystectomy samples  
- **18,000+** demonstration episodes (≈13,000 optimal, ≈5,000 recovery)  
- **17** labeled surgical tasks per episode (with optimal and recovery executions)
- **Multi-view video**: stereo endoscope + two wrist cameras (PSM1 & PSM2)  
- **Kinematics**: Poses & joint states for PSM1, PSM2, PSM3 & ECM  

---

## 🚀 Quickstart

1. **Folder Structure** Create empty `ImitateCholec` folder and subfolders `data_collector_1` and `data_collector_2`. 
2. **Download** the individual tissue zip files from the [JHU Research Data Repository](https://doi.org/10.7281/T1PF3FYK) and save into the folder structure (see [Dataset folder structure](#-dataset-folder-structure)).  
3. **Unzip** the downloaded tissue zip files.
4. **Install** ImitateCholec Python package:
   ```bash
   pip install -e .
   ```
5. **Configure** environment variables in `ImitateCholec.env`:
   ```bash
   # Edit ImitateCholec.env and set:
   DATASET_PATH=/path/to/your/data/folder  # Path where you unzipped the dataset
   OUTPUTS_PATH=/path/to/outputs           # Path for generated outputs (videos, images, etc.)
   ```
   > **Note:** Environment variables will be automatically loaded within the scripts.

6. **Load** in PyTorch:
   - For **imitation learning** samples: use `imitation_learning_dataset.py`
   - For **workflow analysis** with surgical task labels: use `workflow_analysis_dataset.py`
   
   ```python
   from ImitateCholec.data_loading.imitation_learning_dataset import ImitationLearningDataset
   # or
   from ImitateCholec.data_loading.workflow_analysis_dataset import CholecWorkflowAnalysisDataset
   ```

---

## 📁 Dataset folder structure

![Dataset folder structure](images/dataset_folder_structure.svg)

```
data/
└── collector_<c>/           # data collector 1 or 2
    └── tissue_<i>/          # porcine sample
        └── task_<name>/     # 17 surgical tasks (e.g., 1_grasping_gallbladder or 2_clipping_first_clip_left_tube)
            └── demo_<timestamp>/  # demonstration episode
                ├── da_vinci_stereo_left/   # endoscope left frames
                ├── da_vinci_stereo_right/  # endoscope right frames
                ├── endo_psm1/              # wrist camera PSM1 frames
                ├── endo_psm2/              # wrist camera PSM2 frames
                └── kinematics.csv          # synchronized kinematic logs
```

---

## ⚖️ License

- **Data**: CC BY-NC-SA 4.0  
- **Code**: MIT License
