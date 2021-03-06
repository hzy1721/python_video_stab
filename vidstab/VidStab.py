"""VidStab: a class for stabilizing video files"""

from .cv2_utils import safe_import_cv2
safe_import_cv2()  # inform user of pip install vidstab[cv2] if ModuleNotFoundError

import os
import time
import warnings
import cv2
import numpy as np
import imutils
import imutils.feature.factories as kp_factory
from . import general_utils
from . import vidstab_utils
from . import border_utils
from . import auto_border_utils
from . import plot_utils
from .frame_queue import FrameQueue
from . frame import Frame


class VidStab:
    """A class for stabilizing video files

    The VidStab class can be used to stabilize videos using functionality from OpenCV.
    Input video is read from file, put through stabilization process, and written to
    an output file.

    The process calculates optical flow (``cv2.calcOpticalFlowPyrLK``) from frame to frame using
    keypoints generated by the keypoint method specified by the user.  The optical flow will
    be used to generate frame to frame transformations (``cv2.estimateRigidTransform``).
    Transformations will be applied (``cv2.warpAffine``) to stabilize video.

    This class is based on the `work presented by Nghia Ho <http://nghiaho.com/?p=2093>`_

    :param kp_method: String of the type of keypoint detector to use. Available options are:
                        ``["GFTT", "BRISK", "DENSE", "FAST", "HARRIS", "MSER", "ORB", "STAR"]``.
                        ``["SIFT", "SURF"]`` are additional non-free options available depending
                        on your build of OpenCV.  The non-free detectors are not tested with this package.
    :param processing_max_dim: Working with large frames can harm performance (especially in live video).
                                   Setting this parameter can restrict frame size while processing.
                                   The outputted frames will remain the original size.

                                   For example:

                                   * If an input frame shape is `(200, 400, 3)` and `processing_max_dim` is
                                     100.  The frame will be resized to `(50, 100, 3)` before processing.

                                   * If an input frame shape is `(400, 200, 3)` and `processing_max_dim` is
                                     100.  The frame will be resized to `(100, 50, 3)` before processing.

                                   * If an input frame shape is `(50, 50, 3)` and `processing_max_dim` is
                                     100.  The frame be unchanged for processing.

    :param args: Positional arguments for keypoint detector.
    :param kwargs: Keyword arguments for keypoint detector.

    :ivar kp_method: a string naming the keypoint detector being used
    :ivar processing_max_dim: max image dimension while processing transforms
    :ivar kp_detector: the keypoint detector object being used
    :ivar trajectory: a 2d numpy array showing the trajectory of the input video
    :ivar smoothed_trajectory: a 2d numpy array showing the smoothed trajectory of the input video
    :ivar transforms: a 2d numpy array storing the transformations used from frame to frame
    """

    def __init__(self, kp_method='GFTT', processing_max_dim=float('inf'), *args, **kwargs):
        """instantiate VidStab class

        :param kp_method: String of the type of keypoint detector to use. Available options are:
                        ``["GFTT", "BRISK", "DENSE", "FAST", "HARRIS", "MSER", "ORB", "STAR"]``.
                        ``["SIFT", "SURF"]`` are additional non-free options available depending
                        on your build of OpenCV.  The non-free detectors are not tested with this package.
        :param processing_max_dim: Working with large frames can harm performance (especially in live video).
                                   Setting this parameter can restrict frame size while processing.
                                   The outputted frames will remain the original size.

                                   For example:
                                     * If an input frame shape is `(200, 400, 3)` and `processing_max_dim` is
                                   100.  The frame will be resized to `(50, 100, 3)` before processing.
                                     * If an input frame shape is `(400, 200, 3)` and `processing_max_dim` is
                                   100.  The frame will be resized to `(100, 50, 3)` before processing.
                                     * If an input frame shape is `(50, 50, 3)` and `processing_max_dim` is
                                   100.  The frame be unchanged for processing.

        :param args: Positional arguments for keypoint detector.
        :param kwargs: Keyword arguments for keypoint detector.
        """

        # 角点检测方法字符串
        self.kp_method = kp_method
        # use original defaults in http://nghiaho.com/?p=2093 if GFTT with no additional (kw)args
        # 检测方法是GFTT，没有其他参数
        if kp_method == 'GFTT' and args == () and kwargs == {}:
            # 用默认参数创建检测器
            self.kp_detector = kp_factory.FeatureDetector_create('GFTT',            # 方法：GFTT
                                                                 maxCorners=200,    # 最大角点数：200
                                                                 qualityLevel=0.01, # 最低可接受的角点质量：0.01
                                                                 minDistance=30.0,  # 角点之间最小欧式距离：30
                                                                 blockSize=3)       # 计算M用到的块大小：3
        else:
            # 传入参数创建检测器
            self.kp_detector = kp_factory.FeatureDetector_create(kp_method, *args, **kwargs)

        # 最长边的长度，另一条边等比例缩小或不变
        self.processing_max_dim = processing_max_dim
        # 缩放参数
        self._processing_resize_kwargs = {}

        #
        self._smoothing_window = 30
        # [dx, dy, da]的原始变换列表
        self._raw_transforms = []
        # 相对于第一帧的累积变换列表
        self._trajectory = []
        # 轨迹 平滑轨迹 变换 numpy array
        self.trajectory = self.smoothed_trajectory = self.transforms = None

        # 帧队列
        self.frame_queue = FrameQueue()
        # 前一帧的角点和灰度图
        self.prev_kps = self.prev_gray = None

        # VideoWriter
        self.writer = None

        # 层选项
        self.layer_options = {
            # 层函数
            'layer_func': None,
            # 前一帧
            'prev_frame': None
        }

        # 边框选项
        self.border_options = {}
        # 自动设置边框
        self.auto_border_flag = False
        # 极端帧角点
        self.extreme_frame_corners = {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0}
        # 帧角点
        self.frame_corners = None

        # 默认去抖动帧输出
        self._default_stabilize_frame_output = None

    def _resize_frame(self, frame):
        # 缩放参数为空
        if self._processing_resize_kwargs == {}:
            # 用户提供了最大边长
            if self.processing_max_dim:
                # 帧的高、宽
                shape = frame.shape
                # 当前最大边长
                max_dim_size = max(shape)

                # <=的话，直接使用当前最大边长
                if max_dim_size <= self.processing_max_dim:
                    self.processing_max_dim = max_dim_size
                    self._processing_resize_kwargs = None
                # >的话，需要缩小到用户提供的最大边长
                else:
                    max_dim_ind = shape.index(max_dim_size)
                    max_dim_name = ['height', 'width'][max_dim_ind]
                    self._processing_resize_kwargs = {max_dim_name: self.processing_max_dim}

        # 没有缩放参数：直接返回
        if self._processing_resize_kwargs is None:
            return frame

        # 缩小
        resized = imutils.resize(frame, **self._processing_resize_kwargs)
        return resized

    def _update_prev_frame(self, current_frame_gray):
        # 更新前一帧灰度图
        self.prev_gray = current_frame_gray[:]
        # 更新前一帧角点
        self.prev_kps = self.kp_detector.detect(self.prev_gray)
        # noinspection PyArgumentList
        # 转换格式
        self.prev_kps = np.array([kp.pt for kp in self.prev_kps], dtype='float32').reshape(-1, 1, 2)

    def _update_trajectory(self, transform):
        # 轨迹为空
        if not self._trajectory:
            # 直接添加到列表
            self._trajectory.append(transform[:])
        else:
            # gen cumsum for new row and append
            # 累加变换[dx, dy, da]，也就是说_trajectory中的变换都是相对于第一帧的
            self._trajectory.append([self._trajectory[-1][j] + x for j, x in enumerate(transform)])

    def _gen_next_raw_transform(self):
        # 取当前帧
        current_frame = self.frame_queue.frames[-1]
        # 当前帧的灰度图
        current_frame_gray = current_frame.gray_image
        # 缩放后的灰度图
        current_frame_gray = self._resize_frame(current_frame_gray)

        # calc flow of movement
        # 计算光流
        optical_flow = cv2.calcOpticalFlowPyrLK(self.prev_gray,         # 前一帧的灰度图
                                                current_frame_gray,     # 当前帧的灰度图
                                                self.prev_kps, None)    # 前一帧的角点

        # (cur_matched_kp, prev_matched_kp)
        matched_keypoints = vidstab_utils.match_keypoints(optical_flow, self.prev_kps)
        # [dx, dy, da]
        transform_i = vidstab_utils.estimate_partial_transform(matched_keypoints)

        # update previous frame info for next iteration
        # 更新前一帧灰度图和角点
        self._update_prev_frame(current_frame_gray)
        # 添加到原始变换列表
        self._raw_transforms.append(transform_i[:])
        # 更新轨迹列表：存储相对于第一帧的变换
        self._update_trajectory(transform_i)

    def _init_is_complete(self, gen_all):
        # 生成所有帧的变换
        if gen_all:
            return False

        max_ind = min([self.frame_queue.max_frames,
                       self.frame_queue.max_len])

        # 处理的帧数超过了max_frames或max_len：可以完成了
        if self.frame_queue.inds[-1] >= max_ind - 1:
            return True

        # 未完成
        return False

    def _process_first_frame(self, array=None):
        # read first frame
        # 读取一帧并添加到帧队列中
        _, _, _ = self.frame_queue.read_frame(array=array, pop_ind=False)

        # 参数未提供帧数组，帧队列为空：没有可用的帧
        if array is None and len(self.frame_queue.frames) == 0:
            raise ValueError('First frame is None. Check if input file/stream is correct.')

        # convert to gray scale
        # 取第一帧
        prev_frame = self.frame_queue.frames[-1]
        # 第一帧的灰度图
        prev_frame_gray = prev_frame.gray_image
        # 把灰度图按processing_max_dim缩放
        prev_frame_gray = self._resize_frame(prev_frame_gray)

        # detect keypoints
        # 检测第一帧中的角点keypoints
        prev_kps = self.kp_detector.detect(prev_frame_gray)
        # noinspection PyArgumentList
        # 第一帧中的角点
        self.prev_kps = np.array([kp.pt for kp in prev_kps], dtype='float32').reshape(-1, 1, 2)

        # prev_frame_bgra = prev_frame.bgra_image
        # for kp in self.prev_kps:
        #     kpx, kpy = kp[0][0], kp[0][1]
        #     cv2.circle(prev_frame_bgra, (kpx, kpy), 1, (0, 0, 255), -1)
        # cv2.imwrite('keypoint/first.jpg', prev_frame_bgra)

        # 第一帧的灰度图
        self.prev_gray = prev_frame_gray[:]

    def _init_trajectory(self, smoothing_window, max_frames, gen_all=False, show_progress=False):
        # 平滑窗口
        self._smoothing_window = smoothing_window

        # max_frames默认为inf
        if max_frames is None:
            max_frames = float('inf')

        # 视频总帧数
        frame_count = self.frame_queue.source_frame_count
        # 初始化进度条，返回IncrementalBar实例
        bar = general_utils.init_progress_bar(frame_count, max_frames, show_progress, gen_all)

        # 处理第一帧，得到第一帧的角点和灰度图
        self._process_first_frame()

        # iterate through frames
        while True:
            # read current frame
            # 读下一帧并保存到帧队列
            _, _, break_flag = self.frame_queue.read_frame(pop_ind=False)
            # 没有帧了
            if not self.frame_queue.grabbed_frame:
                # 进度+1
                general_utils.update_progress_bar(bar, show_progress)
                break

            # 计算当前帧的灰度图，得到相邻两帧的光流和原始变换，添加到原始变换列表和累积变换列表，并更新前一帧的灰度图和角点
            self._gen_next_raw_transform()

            # 完成初始化
            if self._init_is_complete(gen_all):
                break

            # 进度+1
            general_utils.update_progress_bar(bar, show_progress)

        # 做平滑，并只保留前max_frames-1帧
        self._gen_transforms()

        # 返回进度条实例
        return bar

    def _init_writer(self, output_path, frame_shape, output_fourcc, fps):
        # set output and working dim
        # 高、宽
        h, w = frame_shape

        # setup video writer
        self.writer = cv2.VideoWriter(output_path,
                                      cv2.VideoWriter_fourcc(*output_fourcc),
                                      fps, (w, h), True)

    def _set_border_options(self, border_size, border_type):
        # 功能性border_size和neg_border_size
        functional_border_size, functional_neg_border_size = border_utils.functional_border_sizes(border_size)

        # 边框相关的设置：border_options
        self.border_options = {
            # 边框类型
            'border_type': border_type,
            # 边框宽度
            'border_size': functional_border_size,
            # "负的"边框宽度
            'neg_border_size': functional_neg_border_size,
            # 末端帧角点
            'extreme_frame_corners': self.extreme_frame_corners,
            # 是否自动设置边框
            'auto_border_flag': self.auto_border_flag
        }

    def _apply_transforms(self, output_path, max_frames, use_stored_transforms,
                          output_fourcc='MJPG', border_type='black', border_size=0,
                          layer_func=None, playback=False, progress_bar=None):

        # 设置边框选项：self.border_options
        self._set_border_options(border_size, border_type)
        # 设置层函数
        self.layer_options['layer_func'] = layer_func

        while True:
            # 进度+1
            general_utils.update_progress_bar(progress_bar)
            # 读取一帧
            i, frame_i, break_flag = self.frame_queue.read_frame()

            # 没有帧可以处理或处理完成
            if not self.frame_queue.frames_to_process() or break_flag:
                break

            # 不使用保存的变换
            if not use_stored_transforms:
                # 生成一帧的原始变换
                self._gen_next_raw_transform()

            # 对一帧应用变换
            transformed = self._apply_next_transform(i, frame_i, use_stored_transforms=use_stored_transforms)

            # 应用变换后的的帧为空
            if transformed is None:
                warnings.warn('Video is longer than available transformations; halting process.')
                break

            #
            break_playback = general_utils.playback_video(transformed, playback,
                                                          delay=min([self._smoothing_window, max_frames]))
            # 无法实时演示：退出
            if break_playback:
                break

            # VideoWriter为空
            if self.writer is None:
                # 初始化VideoWriter
                self._init_writer(output_path, transformed.shape[:2], output_fourcc,
                                  fps=self.frame_queue.source_fps)

            # 写入一帧
            self.writer.write(transformed)

        # 释放VideoWriter
        self.writer.release()
        self.writer = None
        # 进度完成
        general_utils.update_progress_bar(progress_bar, finish=True)
        # 关闭窗口
        cv2.destroyAllWindows()

    def _gen_transforms(self):
        # 累积变换的numpy array
        self.trajectory = np.array(self._trajectory)
        # 平滑后的累积变换
        self.smoothed_trajectory = general_utils.bfill_rolling_mean(self.trajectory, n=self._smoothing_window)
        # 平滑后的原始变换
        self.transforms = np.array(self._raw_transforms) + (self.smoothed_trajectory - self.trajectory)

        # Dump superfluous frames
        # noinspection PyProtectedMember
        # max_frames不为空
        n = self.frame_queue._max_frames
        if n:
            # 只保留前n-1帧
            # 累积变换
            self.trajectory = self.trajectory[:n - 1, :]
            # 平滑后的累积变换
            self.smoothed_trajectory = self.smoothed_trajectory[:n - 1, :]
            # 平滑后的原始变换
            self.transforms = self.transforms[:n - 1, :]

    def gen_transforms(self, input_path, smoothing_window=30, show_progress=True):
        """Generate stabilizing transforms for a video

        This method will populate the following instance attributes: trajectory, smoothed_trajectory, & transforms.
        The resulting transforms can subsequently be used for video stabilization by using ``VidStab.apply_transforms``
        or ``VidStab.stabilize`` with ``use_stored_transforms=True``.

        :param input_path: Path to input video to stabilize.
                           Will be read with ``cv2.VideoCapture``; see opencv documentation for more info.
        :param smoothing_window: window size to use when smoothing trajectory
        :param show_progress: Should a progress bar be displayed to console?
        :return: Nothing; this method populates attributes of VidStab objects

        >>> from vidstab.VidStab import VidStab
        >>> stabilizer = VidStab()
        >>> stabilizer.gen_transforms(input_path='input_video.mov')
        >>> stabilizer.apply_transforms(input_path='input_video.mov', output_path='stable_video.avi')
        """
        # 设置平滑窗口大小
        self._smoothing_window = smoothing_window

        # 输入文件不存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'{input_path} does not exist')

        # 设置帧队列来源
        self.frame_queue.set_frame_source(cv2.VideoCapture(input_path))
        # 设置帧队列的max_len和max_frames
        self.frame_queue.reset_queue(max_len=smoothing_window + 1, max_frames=float('inf'))
        # 生成累积变换
        bar = self._init_trajectory(smoothing_window=smoothing_window,  # 平滑窗口大小
                                    max_frames=float('inf'),            # max_frames
                                    gen_all=True,                       #
                                    show_progress=show_progress)        # 显示进度条

        # 进度+1
        general_utils.update_progress_bar(bar, finish=True)

    def apply_transforms(self, input_path, output_path, output_fourcc='MJPG',
                         border_type='black', border_size=0, layer_func=None, show_progress=True, playback=False):
        """Apply stored transforms to a video and save output to file

        Use the transforms generated by ``VidStab.gen_transforms`` or ``VidStab.stabilize`` in stabilization process.
        This method is a wrapper for ``VidStab.stabilize`` with ``use_stored_transforms=True``;
        it is included for backwards compatibility.

        :param input_path: Path to input video to stabilize.
                           Will be read with ``cv2.VideoCapture``; see opencv documentation for more info.
        :param output_path: Path to save stabilized video.
                            Will be written with ``cv2.VideoWriter``; see opencv documentation for more info.
        :param output_fourcc: FourCC is a 4-byte code used to specify the video codec.
        :param border_type: How to handle negative space created by stabilization translations/rotations.
                            Options: ``['black', 'reflect', 'replicate']``
        :param border_size: Size of border in output.
                            Positive values will pad video equally on all sides,
                            negative values will crop video equally on all sides,
                            ``'auto'`` will attempt to minimally pad to avoid cutting off portions of transformed frames
        :param layer_func: Function to layer frames in output.
                           The function should accept 2 parameters: foreground & background.
                           The current frame of video will be passed as foreground,
                           the previous frame will be passed as the background
                           (after the first frame of output the background will be the output of
                           layer_func on the last iteration)
        :param show_progress: Should a progress bar be displayed to console?
        :param playback: Should the a comparison of input video/output video be played back during process?
        :return: Nothing is returned.  Output of stabilization is written to ``output_path``.

        >>> from vidstab.VidStab import VidStab
        >>> stabilizer = VidStab()
        >>> stabilizer.gen_transforms(input_path='input_video.mov')
        >>> stabilizer.apply_transforms(input_path='input_video.mov', output_path='stable_video.avi')
        """
        self.stabilize(input_path, output_path, smoothing_window=self._smoothing_window, max_frames=float('inf'),
                       border_type=border_type, border_size=border_size, layer_func=layer_func, playback=playback,
                       use_stored_transforms=True, show_progress=show_progress, output_fourcc=output_fourcc)

    def _apply_next_transform(self, i, frame_i, use_stored_transforms=False):
        # 不使用保存的变换
        if not use_stored_transforms:
            # 生成平滑后的原始变换
            self._gen_transforms()

        if i is None:
            # 弹出一个索引
            i = self.frame_queue.inds.popleft()

        if frame_i is None:
            # 弹出一帧
            frame_i = self.frame_queue.frames.popleft()

        try:
            # 取出平滑后的原始变换
            transform_i = self.transforms[i, :]
        except IndexError:
            return None

        # 生成变换后的帧
        transformed = vidstab_utils.transform_frame(frame_i,
                                                    transform_i,
                                                    self.border_options['border_size'],
                                                    self.border_options['border_type'])

        # 后处理
        transformed, self.layer_options = vidstab_utils.post_process_transformed_frame(transformed,
                                                                                       self.border_options,
                                                                                       self.layer_options)

        # 转换颜色方案
        transformed = transformed.cvt_color(frame_i.color_format)

        # 返回应用变换后的帧
        return transformed

    def stabilize_frame(self, input_frame, smoothing_window=30,
                        border_type='black', border_size=0, layer_func=None,
                        use_stored_transforms=False):
        """Stabilize single frame of video being iterated

        Perform video stabilization a single frame at a time.  Outputted stabilized frame will be on a
        ``smoothing_window`` delay.  When frames processed is ``< smoothing_window``, black frames will be returned.
        When frames processed is ``>= smoothing_window``, the stabilized frame ``smoothing_window`` ago will be
        returned.  When ``input_frame is None`` stabilization will still be attempted, if there are not frames left to
        process then ``None`` will be returned.

        :param input_frame: An OpenCV image (as numpy array) or None
        :param smoothing_window: window size to use when smoothing trajectory
        :param border_type: How to handle negative space created by stabilization translations/rotations.
                            Options: ``['black', 'reflect', 'replicate']``
        :param border_size: Size of border in output.
                            Positive values will pad video equally on all sides,
                            negative values will crop video equally on all sides,
                            ``'auto'`` will attempt to minimally pad to avoid cutting off portions of transformed frames
        :param layer_func: Function to layer frames in output.
                           The function should accept 2 parameters: foreground & background.
                           The current frame of video will be passed as foreground,
                           the previous frame will be passed as the background
                           (after the first frame of output the background will be the output of
                           layer_func on the last iteration)
        :param use_stored_transforms: should stored transforms from last stabilization be used instead of
                                      recalculating them?
        :return: 1 of 3 outputs will be returned:

            * Case 1 - Stabilization process is still warming up
                + **An all black frame of same shape as input_frame is returned.**
                + A minimum of ``smoothing_window`` frames need to be processed to perform stabilization.
                + This behavior was based on ``cv2.bgsegm.createBackgroundSubtractorMOG()``.
            * Case 2 - Stabilization process is warmed up and ``input_frame is not None``
                + **A stabilized frame is returned**
                + This will not be the stabilized version of ``input_frame``.
                  Stabilization is on an ``smoothing_window`` frame delay
            * Case 3 - Stabilization process is finished
                + **None**

        >>> from vidstab.VidStab import VidStab
        >>> stabilizer = VidStab()
        >>> vidcap = cv2.VideoCapture('input_video.mov')
        >>> while True:
        >>>     grabbed_frame, frame = vidcap.read()
        >>>     # Pass frame to stabilizer even if frame is None
        >>>     # stabilized_frame will be an all black frame until iteration 30
        >>>     stabilized_frame = stabilizer.stabilize_frame(input_frame=frame,
        >>>                                                   smoothing_window=30)
        >>>     if stabilized_frame is None:
        >>>         # There are no more frames available to stabilize
        >>>         break
        """
        self._set_border_options(border_size, border_type)
        self.layer_options['layer_func'] = layer_func
        self._smoothing_window = smoothing_window

        # Store first frame
        if self.frame_queue.max_len is None:
            self.frame_queue.reset_queue(max_len=smoothing_window + 1, max_frames=float('inf'))

            self._process_first_frame(array=input_frame)

            blank_frame = Frame(np.zeros_like(input_frame))
            blank_frame = border_utils.crop_frame(blank_frame, self.border_options)

            if self.border_options['border_size'] > 0:
                blank_frame_alpha, _ = vidstab_utils.border_frame(blank_frame,
                                                                  self.border_options['border_size'],
                                                                  self.border_options['border_type'])

                blank_frame_np = Frame(blank_frame_alpha).cvt_color(blank_frame.color_format)
                blank_frame = Frame(blank_frame_np)

            self._default_stabilize_frame_output = blank_frame.image

            return self._default_stabilize_frame_output

        if len(self.frame_queue.frames) == 0:
            return None

        frame_i = None
        if input_frame is not None:
            _, frame_i, _ = self.frame_queue.read_frame(array=input_frame, pop_ind=False)
            if not use_stored_transforms:
                self._gen_next_raw_transform()

        if not self._init_is_complete(gen_all=False):
            return self._default_stabilize_frame_output

        stabilized_frame = self._apply_next_transform(self.frame_queue.i,
                                                      frame_i,
                                                      use_stored_transforms=use_stored_transforms)

        return stabilized_frame

    def stabilize(self, input_path, output_path, smoothing_window=30, max_frames=float('inf'),
                  border_type='black', border_size=0, layer_func=None, playback=False,
                  use_stored_transforms=False, show_progress=True, output_fourcc='MJPG'):
        """Read video, perform stabilization, & write stabilized video to file

        :param input_path: Path to input video to stabilize.
                           Will be read with ``cv2.VideoCapture``; see opencv documentation for more info.
        :param output_path: Path to save stabilized video.
                            Will be written with ``cv2.VideoWriter``; see opencv documentation for more info.
        :param smoothing_window: window size to use when smoothing trajectory
        :param max_frames: The maximum amount of frames to stabilize/process.
                           The list of available codes can be found in fourcc.org.
                           See cv2.VideoWriter_fourcc documentation for more info.
        :param border_type: How to handle negative space created by stabilization translations/rotations.
                            Options: ``['black', 'reflect', 'replicate']``
        :param border_size: Size of border in output.
                            Positive values will pad video equally on all sides,
                            negative values will crop video equally on all sides,
                            ``'auto'`` will attempt to minimally pad to avoid cutting off portions of transformed frames
        :param layer_func: Function to layer frames in output.
                           The function should accept 2 parameters: foreground & background.
                           The current frame of video will be passed as foreground,
                           the previous frame will be passed as the background
                           (after the first frame of output the background will be the output of
                           layer_func on the last iteration)
        :param use_stored_transforms: should stored transforms from last stabilization be used instead of
                                      recalculating them?
        :param playback: Should the a comparison of input video/output video be played back during process?
        :param show_progress: Should a progress bar be displayed to console?
        :param output_fourcc: FourCC is a 4-byte code used to specify the video codec.
        :return: Nothing is returned.  Output of stabilization is written to ``output_path``.

        >>> from vidstab.VidStab import VidStab
        >>> stabilizer = VidStab()
        >>> stabilizer.stabilize(input_path='input_video.mov', output_path='stable_video.avi')

        >>> stabilizer = VidStab(kp_method = 'ORB')
        >>> stabilizer.stabilize(input_path='input_video.mov', output_path='stable_video.avi')
        """
        # 重置VideoWriter
        self.writer = None

        # 自动设置边框宽度
        if border_size == 'auto':
            self.auto_border_flag = True

        # 参数错误：input_path不存在，且不是整数(摄像头编号)
        if not os.path.exists(input_path) and not isinstance(input_path, int):
            raise FileNotFoundError(f'{input_path} does not exist')

        # 把视频/摄像头设置为帧的源
        self.frame_queue.set_frame_source(cv2.VideoCapture(input_path))

        # wait for camera to start up
        # 如果使用摄像头，等待0.1s
        if isinstance(input_path, int):
            time.sleep(0.1)

        # 设置帧队列的max_len和max_frames
        self.frame_queue.reset_queue(max_len=smoothing_window + 1, max_frames=max_frames)

        # 自动设置边框，不使用保存的变换
        if self.auto_border_flag and not use_stored_transforms:
            # 使用保存的变换
            use_stored_transforms = True
            # 生成平滑后的原始变换
            self.gen_transforms(input_path, smoothing_window=smoothing_window, show_progress=show_progress)
            # 设置帧来源
            self.frame_queue.set_frame_source(cv2.VideoCapture(input_path))
            # 设置max_len和max_frames
            self.frame_queue.reset_queue(max_len=smoothing_window + 1, max_frames=max_frames)

            # 初始化进度条
            bar = general_utils.init_progress_bar(self.frame_queue.source_frame_count,
                                                  max_frames,
                                                  show_progress)
            # 读取n帧
            self.frame_queue.populate_queue(smoothing_window)

        # 不自动设置边框，不使用保存的变换
        elif not use_stored_transforms:
            # 生成平滑后的原始变换
            bar = self._init_trajectory(smoothing_window, max_frames, show_progress=show_progress)
        # 不自动设置边框，使用保存的变换
        else:
            # 初始化进度条
            bar = general_utils.init_progress_bar(self.frame_queue.source_frame_count,
                                                  max_frames,
                                                  show_progress)
            # 读取n帧
            self.frame_queue.populate_queue(smoothing_window)

        # 自动设置边框
        if self.auto_border_flag:
            # 第1帧
            frame_1 = self.frame_queue.frames[0]
            # 极端帧角点
            self.extreme_frame_corners = auto_border_utils.extreme_corners(frame_1.image, self.transforms)
            # 最小边框
            border_size = auto_border_utils.min_auto_border_size(self.extreme_frame_corners)

        # 应用变换
        self._apply_transforms(output_path, max_frames, use_stored_transforms=use_stored_transforms,
                               border_type=border_type, border_size=border_size,
                               layer_func=layer_func, playback=playback,
                               output_fourcc=output_fourcc, progress_bar=bar)

    def plot_trajectory(self):
        """Plot video trajectory

        Create a plot of the video's trajectory & smoothed trajectory.
        Separate subplots are used to show the x and y trajectory.

        :return: tuple of matplotlib objects ``(Figure, (AxesSubplot, AxesSubplot))``

        >>> from vidstab import VidStab
        >>> import matplotlib.pyplot as plt
        >>> stabilizer = VidStab()
        >>> stabilizer.gen_transforms(input_path='input_video.mov')
        >>> stabilizer.plot_trajectory()
        >>> plt.show()
        """
        return plot_utils.plot_trajectory(self.transforms, self.trajectory, self.smoothed_trajectory)

    def plot_transforms(self, radians=False):
        """Plot stabilizing transforms

        Create a plot of the transforms used to stabilize the input video.
        Plots x & y transforms (dx & dy) in a separate subplot than angle transforms (da).

        :param radians: Should angle transforms be plotted in radians?  If ``false``, transforms are plotted in degrees.
        :return: tuple of matplotlib objects ``(Figure, (AxesSubplot, AxesSubplot))``

        >>> from vidstab import VidStab
        >>> import matplotlib.pyplot as plt
        >>> stabilizer = VidStab()
        >>> stabilizer.gen_transforms(input_path='input_video.mov')
        >>> stabilizer.plot_transforms()
        >>> plt.show()
        """
        return plot_utils.plot_transforms(self.transforms, radians)
