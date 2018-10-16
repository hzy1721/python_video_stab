"""VidStab: a class for stabilizing video files"""

try:
    import cv2
except ModuleNotFoundError:
    print("""
    No python bindings for OpenCV found when attempting to `import cv2`.
    If you have not installed OpenCV you can install with:
        
        pip install vidstab[cv2]
        
    If you'd prefer to install OpenCV from source you can see the docs here:
        https://docs.opencv.org/3.4.1/da/df6/tutorial_py_table_of_contents_setup.html
    """)
    raise

import time
from collections import deque
import numpy as np
import imutils
import imutils.feature.factories as kp_factory
import matplotlib.pyplot as plt
from . import general_utils
from . import vidstab_utils


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
    :param args: Positional arguments for keypoint detector.
    :param kwargs: Keyword arguments for keypoint detector.

    :ivar kp_method: a string naming the keypoint detector being used
    :ivar kp_detector: the keypoint detector object being used
    :ivar trajectory: a 2d showing the trajectory of the input video
    :ivar smoothed_trajectory: a 2d numpy array showing the smoothed trajectory of the input video
    :ivar transforms: a 2d numpy array storing the transformations used from frame to frame
    """

    def __init__(self, kp_method='GFTT', *args, **kwargs):
        """instantiate VidStab class

        :param kp_method: String of the type of keypoint detector to use. Available options are:
                        ``["GFTT", "BRISK", "DENSE", "FAST", "HARRIS", "MSER", "ORB", "STAR"]``.
                        ``["SIFT", "SURF"]`` are additional non-free options available depending
                        on your build of OpenCV.  The non-free detectors are not tested with this package.
        :param args: Positional arguments for keypoint detector.
        :param kwargs: Keyword arguments for keypoint detector.
        """

        self.kp_method = kp_method
        # use original defaults in http://nghiaho.com/?p=2093 if GFTT with no additional (kw)args
        if kp_method == 'GFTT' and args == () and kwargs == {}:
            self.kp_detector = kp_factory.FeatureDetector_create('GFTT',
                                                                 maxCorners=200,
                                                                 qualityLevel=0.01,
                                                                 minDistance=30.0,
                                                                 blockSize=3)
        else:
            self.kp_detector = kp_factory.FeatureDetector_create(kp_method, *args, **kwargs)

        self._smoothing_window = None
        self._raw_transforms = []
        self._trajectory = []
        self.trajectory = None
        self.smoothed_trajectory = None
        self.transforms = None
        self.frame_queue = None
        self.frame_queue_inds = None
        self.prev_kps = None
        self.prev_gray = None
        self.vid_cap = None
        self.writer = None

        self.auto_border_flag = False
        self.extreme_frame_corners = {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0}
        self.frame_corners = None

    def _update_prev_frame(self, current_frame_gray):
        self.prev_gray = current_frame_gray[:]
        self.prev_kps = self.kp_detector.detect(self.prev_gray)
        self.prev_kps = np.array([kp.pt for kp in self.prev_kps], dtype='float32').reshape(-1, 1, 2)

    def _update_trajectory(self, transform):
        if not self._trajectory:
            self._trajectory.append(transform[:])
        else:
            # gen cumsum for new row and append
            self._trajectory.append([self._trajectory[-1][j] + x for j, x in enumerate(transform)])

    def _set_extreme_corners(self, frame):
        h, w = frame.shape[:2]
        frame_corners = np.array([[0, 0],  # top left
                                  [h - 1, 0],  # bottom left
                                  [0, w - 1],  # top right
                                  [h - 1, w - 1]],  # bottom right
                                 dtype='float32')
        frame_corners = np.array([frame_corners])

        min_x = min_y = max_x = max_y = 0
        for i in range(self.transforms.shape[0]):
            transform = self.transforms[i, :]
            transform_mat = vidstab_utils.build_transformation_matrix(transform)
            transformed_frame_corners = cv2.transform(frame_corners, transform_mat)

            abs_delta_corners = abs(frame_corners - transformed_frame_corners)

            y_corners = abs_delta_corners[0][:, 0].tolist()
            x_corners = abs_delta_corners[0][:, 1].tolist()
            min_x = max([min_x] + x_corners[:2])
            min_y = max([min_y] + y_corners[::2])
            max_x = max([max_x] + x_corners[2:])
            max_y = max([max_y] + y_corners[1::2])

        self.extreme_frame_corners = {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}

    def _populate_queues(self, smoothing_window, max_frames):
        n = min([smoothing_window, max_frames])

        for i in range(n):
            grabbed_frame, frame = self.vid_cap.read()
            if not grabbed_frame:
                break

            self.frame_queue.append(frame)
            self.frame_queue_inds.append(i)

    def _gen_next_raw_transform(self):
        current_frame_gray = cv2.cvtColor(self.frame_queue[-1], cv2.COLOR_BGR2GRAY)

        # calc flow of movement
        optical_flow = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                current_frame_gray,
                                                self.prev_kps, None)

        matched_keypoints = vidstab_utils.match_keypoints(optical_flow, self.prev_kps)
        transform_i = vidstab_utils.estimate_partial_transform(matched_keypoints)

        # update previous frame info for next iteration
        self._update_prev_frame(current_frame_gray)
        self._raw_transforms.append(transform_i[:])
        self._update_trajectory(transform_i)

    def _init_trajectory(self, smoothing_window, max_frames, gen_all=False, show_progress=False):
        frame_count = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if gen_all:
            message = 'Generating Transforms'
        else:
            message = 'Stabilizing'
        bar = general_utils.init_progress_bar(frame_count, max_frames, show_progress, message)

        # read first frame
        grabbed_frame, prev_frame = self.vid_cap.read()
        # convert to gray scale
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        # detect keypoints
        prev_kps = self.kp_detector.detect(prev_frame_gray)
        self.prev_kps = np.array([kp.pt for kp in prev_kps], dtype='float32').reshape(-1, 1, 2)

        # store frame
        self.frame_queue.append(prev_frame)
        self.prev_gray = prev_frame_gray[:]

        if max_frames is None:
            max_frames = float('inf')

        # iterate through frames count
        grabbed_frame = True

        while grabbed_frame:
            # read current frame
            grabbed_frame, cur_frame = self.vid_cap.read()
            if not grabbed_frame:
                if show_progress and bar is not None:
                    bar.next()
                break

            self.frame_queue.append(cur_frame)
            if not self.frame_queue_inds:
                self.frame_queue_inds.append(0)
            else:
                self.frame_queue_inds.append(self.frame_queue_inds[-1] + 1)
            self._gen_next_raw_transform()

            if not gen_all:
                if (self.frame_queue_inds[-1] >= max_frames - 1 or
                        self.frame_queue_inds[-1] >= smoothing_window - 1):
                    break

            if show_progress and bar is not None:
                bar.next()

        self._gen_transforms(smoothing_window)

        return bar

    def _init_writer(self, output_path, frame_shape, output_fourcc, fps):
        # set output and working dims
        h, w = frame_shape

        # setup video writer
        self.writer = cv2.VideoWriter(output_path,
                                      cv2.VideoWriter_fourcc(*output_fourcc),
                                      fps, (w, h), True)

    def _apply_transforms(self, output_path, max_frames, smoothing_window, output_fourcc='MJPG',
                          border_type='black', border_size=0, layer_func=None, playback=False, progress_bar=None):

        if border_type not in ['black', 'reflect', 'replicate']:
            raise ValueError('Invalid border type')

        if border_size < 0:
            neg_border_size = 100 + abs(border_size)
            border_size = 100
        else:
            neg_border_size = 0

        prev_frame = self.frame_queue.popleft()
        (h, w) = prev_frame.shape[:2]
        h += 2 * border_size
        w += 2 * border_size

        # initialize transformation matrix
        grabbed_frame = True
        while len(self.frame_queue) > 0 or grabbed_frame:
            if progress_bar:
                progress_bar.next()

            grabbed_frame, next_frame = self.vid_cap.read()
            if grabbed_frame:
                self.frame_queue.append(next_frame)
                self.frame_queue_inds.append(self.frame_queue_inds[-1] + 1)
                self._gen_next_raw_transform()
                self._gen_transforms(smoothing_window=smoothing_window)

            i = self.frame_queue_inds.popleft()
            frame_i = self.frame_queue.popleft()
            transform_i = self.transforms[i, :]

            if i >= max_frames:
                break

            # build transformation matrix
            transform = vidstab_utils.build_transformation_matrix(transform_i)

            bordered_frame, border_mode = vidstab_utils.border_frame(frame_i, border_size, border_type)

            transformed = cv2.warpAffine(bordered_frame,
                                         transform,
                                         bordered_frame.shape[:2][::-1],
                                         borderMode=border_mode)

            buffer = border_size + neg_border_size

            if self.auto_border_flag:
                auto_x = round(1.2 * (buffer - self.extreme_frame_corners['min_x']))
                auto_y = round(1.2 * (buffer - self.extreme_frame_corners['min_y']))
                auto_w = round(1.2 * (frame_i.shape[1] + self.extreme_frame_corners['max_x']))
                auto_h = round(1.2 * (frame_i.shape[0] + self.extreme_frame_corners['max_y']))
                transformed = transformed[auto_y:auto_y + auto_h, auto_x:auto_x + auto_w]

            if layer_func is not None:
                if i > 1:
                    transformed = layer_func(transformed, prev_frame)

                prev_frame = transformed[:]

            if playback:
                resized_transformed = imutils.resize(transformed, width=min([frame_i.shape[0], 1000]))
                playback_frame = resized_transformed

                cv2.imshow('VidStab Playback ({} frame delay if using live video;'
                           ' press Q or ESC to quit)'.format(min([smoothing_window,
                                                                 max_frames])),
                           playback_frame)
                key = cv2.waitKey(1)

                if key == ord("q") or key == 27:
                    break

            if self.writer is None:
                self._init_writer(output_path, transformed.shape[:2], output_fourcc,
                                  fps=int(self.vid_cap.get(cv2.CAP_PROP_FPS)))

            # write frame to output video
            self.writer.write(transformed)

        self.writer.release()
        if progress_bar:
            progress_bar.next()
            progress_bar.finish()

    def _gen_transforms(self, smoothing_window):
        self.trajectory = np.array(self._trajectory)
        self.smoothed_trajectory = general_utils.bfill_rolling_mean(self.trajectory, n=smoothing_window)
        self.transforms = np.array(self._raw_transforms) + (self.smoothed_trajectory - self.trajectory)

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
        self._smoothing_window = smoothing_window
        self.vid_cap = cv2.VideoCapture(input_path)
        self.frame_queue = deque(maxlen=smoothing_window)
        self.frame_queue_inds = deque(maxlen=smoothing_window)
        bar = self._init_trajectory(smoothing_window=smoothing_window,
                                    max_frames=float('inf'),
                                    gen_all=True,
                                    show_progress=show_progress)

        if bar:
            bar.finish()

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
                            ``'auto'`` will attempt to minimally pad to avoid cutting off portions of transformed frames.
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
                            ``'auto'`` will attempt to minimally pad to avoid cutting off portions of transformed frames.
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
        if border_size == 'auto':
            self.auto_border_flag = True

        self.vid_cap = cv2.VideoCapture(input_path)
        frame_count = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # wait for camera to start up
        if isinstance(input_path, int):
            time.sleep(0.1)

        self.frame_queue = deque(maxlen=smoothing_window)
        self.frame_queue_inds = deque(maxlen=smoothing_window)

        if self.auto_border_flag and not use_stored_transforms:
            self.gen_transforms(input_path, smoothing_window=smoothing_window, show_progress=show_progress)
            self.vid_cap = cv2.VideoCapture(input_path)
            self.frame_queue = deque(maxlen=smoothing_window)
            self.frame_queue_inds = deque(maxlen=smoothing_window)

            bar = general_utils.init_progress_bar(frame_count, max_frames, show_progress)
            self._populate_queues(smoothing_window, max_frames)

        elif not use_stored_transforms:
            bar = self._init_trajectory(smoothing_window, max_frames, show_progress=show_progress)
        else:
            bar = general_utils.init_progress_bar(frame_count, max_frames, show_progress)
            self._populate_queues(smoothing_window, max_frames)

        if self.auto_border_flag:
            self._set_extreme_corners(self.frame_queue[0])
            border_size = round(max(self.extreme_frame_corners.values()))

        self._apply_transforms(output_path, max_frames, smoothing_window,
                               border_type=border_type, border_size=border_size, layer_func=layer_func,
                               playback=playback, output_fourcc=output_fourcc, progress_bar=bar)

        cv2.destroyAllWindows()

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
        if self.transforms is None:
            raise AttributeError('No trajectory to plot. '
                                 'Use methods: gen_transforms or stabilize to generate the trajectory attributes')

        with plt.style.context('ggplot'):
            fig, (ax1, ax2) = plt.subplots(2, sharex='all')

            # x trajectory
            ax1.plot(self.trajectory[:, 0], label='Trajectory')
            ax1.plot(self.smoothed_trajectory[:, 0], label='Smoothed Trajectory')
            ax1.set_ylabel('dx')

            # y trajectory
            ax2.plot(self.trajectory[:, 1], label='Trajectory')
            ax2.plot(self.smoothed_trajectory[:, 1], label='Smoothed Trajectory')
            ax2.set_ylabel('dy')

            handles, labels = ax2.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')

            plt.xlabel('Frame Number')

            fig.suptitle('Video Trajectory', x=0.15, y=0.96, ha='left')
            fig.canvas.set_window_title('Trajectory')

            return fig, (ax1, ax2)

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
        if self.transforms is None:
            raise AttributeError('No transforms to plot. '
                                 'Use methods: gen_transforms or stabilize to generate the transforms attribute')

        with plt.style.context('ggplot'):
            fig, (ax1, ax2) = plt.subplots(2, sharex='all')

            ax1.plot(self.transforms[:, 0], label='delta x', color='C0')
            ax1.plot(self.transforms[:, 1], label='delta y', color='C1')
            ax1.set_ylabel('Delta Pixels', fontsize=10)

            if radians:
                ax2.plot(self.transforms[:, 2], label='delta angle', color='C2')
                ax2.set_ylabel('Delta Radians', fontsize=10)
            else:
                ax2.plot(np.rad2deg(self.transforms[:, 2]), label='delta angle', color='C2')
                ax2.set_ylabel('Delta Degrees', fontsize=10)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            fig.legend(handles1 + handles2,
                       labels1 + labels2,
                       loc='upper right',
                       ncol=1)

            plt.xlabel('Frame Number')

            fig.suptitle('Transformations for Stabilizing', x=0.15, y=0.96, ha='left')
            fig.canvas.set_window_title('Transforms')

            return fig, (ax1, ax2)
