# 🔨 Configurations

We provide the detailed explanations of the configuration file as follows. 

Please put special attention to the parameters in `data.normalization`, `train.workspace` and `deploy.workspace`: make sure that they are correctly set in the **camera** coordinates, or you may obtain meaningless outputs. We recommend expanding the translation range in `data.normalization` by 0.15 - 0.3 meters on both sides of the actual workspace range (`train.workspace`) to accommodate point cloud spatial augmentation.


```yaml
robot_type: dual  # choices: dual [for dual-arm robot], single [for single-arm robot]

data_type: teleop  # choices: teleop, wild [for AirExo-2 system]

data: 
  num_action: 20  # action chunk
  sample_traj: true  # sample waypoints from demonstration trajectory
  enhance_gripper: true  # enhance gripper in the point cloud
  voxel_size: 0.005  # voxel size

  normalization:  # normalization parameters 
    trans_min: [-0.9, -0.5, 0.6]  # translation min
    trans_max: [0.9, 0.75, 1.75]  # translation max
    max_gripper_width: 0.1  # max gripper width
    img_mean: [0.485, 0.456, 0.406]  # image mean
    img_std: [0.229, 0.224, 0.225]  # image std

  aligner:
    img_size_resnet: [360, 640]  # image size for resnet dense encoder
    img_size_dinov2: [252, 448]  # image size for DINOv2 dense encoder
    img_coord_size_resnet: [12, 20]  # image coordinate size for resnet dense encoder
    img_coord_size_dinov2: [18, 32]  # image coordinate size for DINOv2 dense encoder

  vis: false  # visualize the data sample (for debug use)


train:
  workspace:  # workspace setting
    min: [-0.7, -0.3, 0.9]  # workspace min
    max: [0.7, 0.55, 1.55]  # workspace max

  augmentation:  # augmentation settings
    point: true  # whether to enable point cloud augmentation
    aug_trans_min: [-0.2, -0.2, -0.2]  # point cloud augmentation: translation min
    aug_trans_max: [0.2, 0.2, 0.2]  # point cloud augmentation: translation max
    aug_rot_min: [-30, -30, -30]  # point cloud augmentation: rotation min (in degrees)
    aug_rot_max: [30, 30, 30]  # point cloud augmentation: rotation max (in degrees)

    color: true  # whether to enable color augmentation
    aug_color_params: [0.4, 0.4, 0.2, 0.1]  # color augmentation parameters for color jitter
    aug_color_prob: 0.2  # the probability to apply color jitter
  
  lr: 3e-4  # learning rate
  batch_size: 240  # batch size
  num_workers: 12  # number of workers in dataloader
  num_steps: 60000  # number of steps
  save_steps: 2500  # saving step intervals
  num_warmup_steps: 2000  # number of warm-up steps
  seed: 233  # seed for training


deploy:
  workspace:  # workspace setting
    min: [-0.7, -0.3, 0.9]  # workspace min
    max: [0.7, 0.55, 1.55]  # workspace max

  agent:  # agent settings
    left_robot_serial: Rizon4-062077  # left robot serial
    right_robot_serial: Rizon4R-062016  # right robot serial
    left_gripper_port: /dev/ttyUSB0  # left gripper port
    right_gripper_port: /dev/ttyUSB1  # right gripper port
    camera_serial: "105422061350"  # camera serial
  
  max_steps: 300  # max steps
  num_inference_steps: 20  # number of inference steps
  gripper_threshold: 0.02  # gripper threshold
  ensemble_mode: new  # ensemble mode, choices: new, old, avg, act, hato
  seed: 233  # seed for deployment

  vis: false  # whether to visualize the prediction in deployment


model:  # model parameters
  image_enc: dinov2-base  # dense (image) encocder backbone
  cloud_enc_dim: 128  # sparse (cloud) encoder dimension
  image_enc_dim: 128  # dense (image) encoder dimension
  obs_feature_dim: 512  # observation feature dimension
  hidden_dim: 512  # hidden dimension
  nheads: 8  # number of attention heads
  num_attn_layers: 4  # numbe of attention layers
  dim_feedforward: 2048  # feedforward dimension
  dropout: 0.1  # dropout ratio
```