# 🤖 Deployment

1. Please set up the correct deployment parameters in the `deploy` field of the configuration file, following [the configuration guide](CONFIG.md).

2. Add the device support libraries of your own devices in the `device` folder.

3. Modify `eval_agent.py` to accomodate your own device.
   - Implement the `__init__(...)` function to initialize the devices. Move the robot to the ready pose, activate the gripper, start the camera, and flush the camera stream.
   - Implement the `intrinsics` property to return the intrinsics of the camera.
   - (Optional) Implement the `left_ready_pose` and `right_ready_pose` property to define the ready pose of the robot.
   - Implement the `get_global_observation()` function to get the RGB and depth observations from the camera.
   - Implement the `get_proprio()` fucntion to get the proprioception of the dual-arm robot (3d left translation + 6d left rotation + 1d left gripper + 3d right translation + 6d right rotation + 1d right gripper).
   - Implement the `action(...)` function to execute the predicted action.
   - Implement the `stop()` function to stop the devices.