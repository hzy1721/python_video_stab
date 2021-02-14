import cv2
import numpy as np
from .frame import Frame
from .pop_deque import PopDeque


class FrameQueue:
    def __init__(self, max_len=None, max_frames=None):
        #
        self.max_len = max_len
        # max_frames
        self.max_frames = max_frames
        # max_frames
        self._max_frames = None

        # 帧队列
        self.frames = PopDeque(maxlen=max_len)
        # 索引队列
        self.inds = PopDeque(maxlen=max_len)
        #
        self.i = None

        # VideoCapture实例
        self.source = None
        # 视频总帧数
        self.source_frame_count = None
        #
        self.source_fps = 30

        #
        self.grabbed_frame = False

    def reset_queue(self, max_len=None, max_frames=None):
        # 重置max_len
        self.max_len = max_len if max_len is not None else self.max_len
        # 重置max_frames
        self.max_frames = max_frames if max_frames is not None else self.max_frames

        #
        has_max_frames = self.max_frames is not None and not np.isinf(self.max_frames)
        if has_max_frames:
            self._max_frames = self.max_frames

        self.frames = PopDeque(maxlen=max_len)
        self.inds = PopDeque(maxlen=max_len)
        self.i = None

    def set_frame_source(self, source):
        # VideoCapture实例
        if isinstance(source, cv2.VideoCapture):
            # VideoCapture实例
            self.source = source
            # 视频总帧数
            self.source_frame_count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
            # 视频FPS
            self.source_fps = int(source.get(cv2.CAP_PROP_FPS))

            # 如果max_frames提供了值
            has_max_frames = self.max_frames is not None and not np.isinf(self.max_frames)
            # 没有max_frames，视频不为空
            if self.source_frame_count > 0 and not has_max_frames:
                # max_frames等于总帧数
                self._max_frames = self.source_frame_count
            # 有max_frames，总帧数小于max_frames
            elif has_max_frames and self.source_frame_count < self.max_frames:
                # max_frames等于总帧数
                self._max_frames = self.source_frame_count
            # 其余情况保持_max_frames不变
            # 没有max_frames，视频为空：_max_frames == None
            # 有max_frames，总帧数>=max_frames：_max_frames == None
        # 只支持VideoCapture
        else:
            raise TypeError('Not yet support for non cv2.VideoCapture frame source.')

    def read_frame(self, pop_ind=True, array=None):
        # 帧来源是VideoCapture
        if isinstance(self.source, cv2.VideoCapture):
            # 读取一帧
            self.grabbed_frame, frame = self.source.read()
        # 帧是参数array
        else:
            frame = array

        # 把帧添加到帧队列中
        return self._append_frame(frame, pop_ind)

    def _append_frame(self, frame, pop_ind=True):
        # 被弹出的帧
        popped_frame = None
        # 需要添加的帧非空
        if frame is not None:
            # 帧队列：pop_append
            popped_frame = self.frames.pop_append(Frame(frame))
            # 索引队列：increment_append
            self.i = self.inds.increment_append()

        # 需要弹出索引，之前的append操作未弹出索引
        if pop_ind and self.i is None:
            # 弹出一个索引
            self.i = self.inds.popleft()

        # 需要弹出索引，之前的append操作未弹出索引，max_frames非空
        if (pop_ind
                and self.i is not None
                and self.max_frames is not None):
            # 是否停止
            break_flag = self.i >= self.max_frames
        else:
            break_flag = None

        return self.i, popped_frame, break_flag

    def populate_queue(self, smoothing_window):
        n = min([smoothing_window, self.max_frames])

        # 读取n帧
        for i in range(n):
            _, _, _ = self.read_frame(pop_ind=False)
            if not self.grabbed_frame:
                break

    def frames_to_process(self):
        # 有帧可以处理
        return len(self.frames) > 0 or self.grabbed_frame
