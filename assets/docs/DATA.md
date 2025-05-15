# 🛢️ Data Collection

For single-arm robot platform, we apply the data collection process in the <a href="https://rh20t.github.io/">RH20T</a> paper. For dual-arm robot platform, we use <a href="https://airexo.tech/airexo/"><i>AirExo</i></a> for teleoperated data collection and <a href="https://airexo.tech/airexo2/"><b><i>AirExo</i>-2</b></a> for in-the-wild data collection and transformation. Please refer to [the ***AirExo*-2** codebase](https://github.com/AirExo/AirExo-2) for details about data collection.

```
collect_toys
|-- calib/
|   |-- [calib timestamp 1].npy/       # calibration results
|   |-- ...
|   `-- [calib timestamp m].npy/       # similar calibration results  
`-- train/
    |-- [episode identifier 1]
    |   |-- meta.json                  # metadata
    |   |-- cam_[serial_number 1]/    
    |   |   |-- color                  # RGB
    |   |   |   |-- [timestamp 1].png
    |   |   |   |-- [timestamp 2].png
    |   |   |   |-- ...
    |   |   |   `-- [timestamp T].png
    |   |   `-- depth                  # depth
    |   |       |-- [timestamp 1].png
    |   |       |-- [timestamp 2].png
    |   |       |-- ...
    |   |       `-- [timestamp T].png
    |   |-- cam_[serial_number 2]/     # similar camera structure
    |   `-- lowdim/                    # low-dimensional data
    |       |-- [timestamp 1].npy      # robot tcp pose, gripper information and gripper action
    |       |-- [timestamp 2].npy
    |       |-- ...
    |       `-- [timestamp T].npy
    `-- [episode identifier 2]         # similar episode structure
```
