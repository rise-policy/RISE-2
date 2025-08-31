# 📝 Changelog

## [Aug 29, 2025] Fix a bug in Spatial Aligner.

Due to unfixed point numbers of each point cloud in a batch, we apply WeightedSpatialInterpolation to each sample iteratively. However, in the original implementation in paper, MLP layers are also included in this module, which adopts BatchNorm as the normalization function. Multiple forwarding of the same BN during one training step hinders the model stability and leads to failure in some extreme cases. After the fix, the model shows improved performance.

We test the performance of RISE, RISE-2 (paper) and RISE-2 (fixed) using the task "pour balls" which is similar to the setting in [RISE](https://arxiv.org/pdf/2404.12281). The models are deployed using Nvidia RTX 3090, predicting 50 steps in a chunk and executing the first 20 steps with the control frequency of 10Hz. **No trajectory smoothing strategy is employed to clearly compare the raw performance.** The models are tested under 10 scenes. The completion rates (percentage of the balls successfully poured into the bowl) are as follows:

<div align="center">

| Scene ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Average | Stable Execution|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RISE | 1 | 1- | 1- | 1 | 1 | 1 | 1 | 0- | 0- | 1 | 0.8 | 0.6 |
| RISE-2 (paper) | 1 | 1 | 1 | 1 | 1 | 1 | 1- | 1 | 1 | 1 | **1** | 0.9 |
| RISE-2 (fixed) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **1** | **1** |

</div>

Although the completion rates before and after the fix are the same, we observe more unstable executions (inaccurate reaching, sharp transition between two action chunks, etc.) from "RISE-2 (paper)" (clearly unstable scenes are noted as "-" in the table). We select the most unstable scene "scene 7" and test the performance:

<div align="center">

| Attempt ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Average | Stable Execution|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RISE-2 (paper) | 0- | 0- | 1 | 0- | 0- | 1- | 1 | 1- | 0- | 0- | 0.4 | 0.2 |
| RISE-2 (fixed) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **1** | **1** |

</div>

The visualization of scene 7 for two implementations (download from [video](https://github.com/chenxi-wang/materials/blob/master/RISE-2/assets/gifs/rise2_exp_pour_balls.mp4)|[gif](https://github.com/chenxi-wang/materials/blob/master/RISE-2/assets/gifs/rise2_exp_pour_balls.gif) if you cannot load the video here):

<div align="center">    
    <img src="https://github.com/chenxi-wang/materials/blob/master/RISE-2/assets/gifs/rise2_exp_pour_balls.gif", width="480", alt="pour_balls" />
    <br> Comparison of RISE-2 performace before and after the fix.
</div>

From the table and video above, we can clearly observe a more stable execution using the fixed version.

**NOTE: If you have already trained RISE-2 models using the old implementation and have no time for re-training, you could call ``policy.spatial_aligner.interp.train()`` after ``policy.eval()`` during evaluation to temporarily improve the performace, but it is still a bit unstable compared to the fixed version.**
