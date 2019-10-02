import sys
sys.path.append('build/')
import os
from libpydenseflow import TVL1FlowExtractor
import numpy as np
import cv2
import h5py
from joblib import delayed
from joblib import Parallel

class FlowExtractor(object):
    def __init__(self, dev_id, bound=20):
        TVL1FlowExtractor.set_device(dev_id)
        self._et = TVL1FlowExtractor(bound)

    def extract_flow(self, frame_list, new_size=None):
        """
        This function extracts the optical flow and interleave x and y channels
        :param frame_list:
        :return:
        """
        frame_size = frame_list[0].shape[:2]
        rst = self._et.extract_flow([x.tostring() for x in frame_list], frame_size[1], frame_size[0])
        print ('start reshaping')
        n_out = len(rst)
        if new_size is None:
            ret = np.zeros((n_out*2, frame_size[0], frame_size[1]))
            for i in xrange(n_out):
                ret[2*i, :] = np.fromstring(rst[i][0], dtype='uint8').reshape(frame_size)
                ret[2*i+1, :] = np.fromstring(rst[i][1], dtype='uint8').reshape(frame_size)
        else:
            ret = np.zeros((n_out*2, new_size[1], new_size[0]))
            for i in xrange(n_out):
                ret[2*i, :] = cv2.resize(np.fromstring(rst[i][0], dtype='uint8').reshape(frame_size), new_size)
                ret[2*i+1, :] = cv2.resize(np.fromstring(rst[i][1], dtype='uint8').reshape(frame_size), new_size)

        return ret

def save_optical_flow(output_folder, flow_frames):
    try:
        os.mkdir(output_folder)
    except OSError:
        pass
    nframes = len(flow_frames) // 2
    for i in range(nframes):
        out_x = '{0}/x_{1:04d}.jpg'.format(output_folder, i+1)
        out_y = '{0}/y_{1:04d}.jpg'.format(output_folder, i+1)
        cv2.imwrite(out_x, flow_frames[2*i])
        cv2.imwrite(out_y, flow_frames[2*i+1])

def read_frames(v):
    img = cv2.imdecode(v, -1)
    return img

if __name__ == "__main__":
    f = FlowExtractor(dev_id=0)
    source = '/z/home/kylemin/dataset/Kinetics-new/Kinetics-400h/'
    sp = 'train'
    split = 48

    #sp = 'val'
    #split = 16

    #for i in range(split):
    if True:
        i = 12
        with h5py.File(source + sp + '_%d.h5'%(i+1), 'r') as hf:
            videos = hf['videos'][:1]
            labels = hf['labels'][:1]
            shapes = hf['shapes'][:1]
        num_video = len(shapes)

        print ('start decoding')
        for j in range(num_video):
            frame_list = Parallel(n_jobs=50, prefer="threads")(delayed(read_frames)(videos[j][k]) for k in range(shapes[j][0]))
            print ('start extract_flow')
            flow_frames = f.extract_flow(frame_list)
            print ('saving')
            save_optical_flow(source+'flow', flow_frames)
