'''
RealSense Camera.

Author: Hongjie Fang, Jirong Liu.
'''

import numpy as np
import pyrealsense2 as rs


class RealSenseRGBDCamera:
    '''
    RealSense RGB-D Camera.
    '''

    dummy_frames = 20

    def __init__(
        self, 
        serial: str, 
        frame_rate = 30, 
        resolution = (1280, 720),
        enable_emitter = True,
        align = True,
        **kwargs
    ) -> None:
        '''
        Initialization.

        Parameters:
        - serial: str, required, the serial number of the realsense device;
        - frame_rate: int, optional, default: 15, the framerate of the realsense camera;
        - resolution: (int, int), optional, default: (1280, 720), the resolution of the realsense camera;
        - enable_emitter: bool, optional, default: True, whether to enable the emitter;
        - align: bool, optional, default: True, whether align the frameset with the RGB image.
        '''
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial = serial

        # Support L515 depth sensing
        self.is_radar = str.isalpha(serial[0])
        depth_resolution = (1024, 768) if self.is_radar else resolution
        if self.is_radar:
            frame_rate = max(frame_rate, 30)
            self.depth_scale = 4000
        else:
            self.depth_scale = 1000

        # Set up device and stream
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, frame_rate)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, frame_rate)

        # Start pipeline
        pipeline_profile = self.pipeline.start(self.config)

        # Set up emitter
        depth_sensor = pipeline_profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.emitter_enabled, int(enable_emitter))

        # Set up alignment
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.with_align = align

        # Get intrinsic
        color_profile = pipeline_profile.get_stream(rs.stream.color) 
        self.intrinsic = color_profile.as_video_stream_profile().get_intrinsics()

        # Skip dummy frames due to over-exposure
        for _ in range(self.dummy_frames):
            self.get_rgbd_images()
        
    def get_rgb_image(self):
        '''
        Get the RGB image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)
        return int(frames.get_timestamp()), color_image 

    def get_depth_image(self):
        '''
        Get the depth image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        return int(frames.get_timestamp()), depth_image
    
    def get_rgbd_images(self):
        '''
        Get the RGB-D images from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        if self.with_align:
            frames = self.align.process(frames)
        color_image = np.asanyarray(frames.get_color_frame().get_data()).astype(np.uint8)
        depth_image = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.uint16)
        return int(frames.get_timestamp()), color_image, depth_image

    def get_intrinsic(self, return_mat: bool = True):
        if return_mat:
            return np.array([
                [self.intrinsic.fx, 0., self.intrinsic.ppx],
                [0., self.intrinsic.fy, self.intrinsic.ppy],
                [0., 0., 1.]
            ], dtype = np.float32)
        else:
            return self.intrinsic
    
    def stop(self):
        self.pipeline.stop()