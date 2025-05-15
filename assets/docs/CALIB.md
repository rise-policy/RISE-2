# 📷 Calibration Guide

As a 3D policy, we need to determine the transformation matrix between the robot base and the global camera. The calibration files are essential for both policy training and policy deployment. Here we provide sample calibration files for both single-arm and dual-arm platforms. The following fields are required for this codebase:
- `camera_serials`: a list of camera serials;
- `camera_serials_global`: a list of global camera serials;
- (single-arm) `camera_serial_inhand`: the camera serial of the in-hand camera;
- (dual-arm) `camera_serial_inhand_left` and `camera_serial_inhand_right`: the camera serial of the left / right in-hand camera;
- `intrinsics`: the dictionary that contains the 3x3 intrinsics of all cameras;
- (single-arm) `camera_to_robot`: the transformation matrix from the camera coordinate to the robot base coordinate;
- (dual-arm) `camera_to_robot_left` and `camera_to_robot_right`: the 4x4 transformation matrix from the camera coordinate to the left/right robot base coordinate.

The calibration results are stored in a dictionary in a `npy` file, using `np.save(..., allow_pickle = True)`.

Here, we provide an example of dual-arm calibration results.

```
{
    'type': 'robot',
    'camera_serials': ['105422061350', '104122064161', '104122061330'],
    'camera_serials_global': ['105422061350'],
    'camera_serial_inhand_left': '104122064161',
    'camera_serial_inhand_right': '104122061330',
    'intrinsics': {
        '105422061350': 
            array([[912.4466 ,   0.     , 633.4127 ],
                   [  0.     , 911.4704 , 364.21265],
                   [  0.     ,   0.     ,   1.     ]], dtype=float32),
        '104122064161': 
            array([[915.71423,   0.     , 638.86804],
                   [  0.     , 915.29736, 357.55472],
                   [  0.     ,   0.     ,   1.     ]], dtype=float32),
        '104122061330': 
            array([[909.9401 ,   0.     , 626.91187],
                   [  0.     , 909.0405 , 354.72583],
                   [  0.     ,   0.     ,   1.     ]], dtype=float32)
    },
    'camera_to_robot_left': {
        '105422061350': 
            array([[ 0.02733019, -0.99962234,  0.00270202, -0.12419845],
                   [-0.92083234, -0.02622826, -0.38907513,  0.5822531 ],
                   [ 0.38899893,  0.00814523, -0.9212016 ,  0.69930893],
                   [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=float32)
    },
    'camera_to_robot_right': {
        '105422061350': 
            array([[-0.00864751, -0.9971284 ,  0.07523859,  0.16229412],
                   [-0.92715275, -0.02019016, -0.37413925,  0.5794803 ],
                   [ 0.37458393, -0.07299308, -0.9243155 ,  0.70862824],
                   [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=float32)}
    }
```