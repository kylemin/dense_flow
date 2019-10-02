import sys
sys.path.append('build/')
import os
from libpydenseflow import TVL1FlowExtractor
import numpy as np
import cv2
import h5py
from joblib import delayed
from joblib import Parallel
import time
import random

class FlowExtractor(object):
    def __init__(self, dev_id, bound=20):
        TVL1FlowExtractor.set_device(dev_id)
        self._et = TVL1FlowExtractor(bound)

    def extract_flow(self, frame_list):
        frame_size = frame_list[0].shape[:2]
        rst = self._et.extract_flow([x.tostring() for x in frame_list], frame_size[1], frame_size[0])
        n_out = len(rst)

        x = np.zeros((n_out, frame_size[0], frame_size[1]))
        y = np.zeros((n_out, frame_size[0], frame_size[1]))
        for i in xrange(n_out):
            x[i, :] = np.fromstring(rst[i][0], dtype='uint8').reshape(frame_size)
            y[i, :] = np.fromstring(rst[i][1], dtype='uint8').reshape(frame_size)

        return x, y, n_out

def save_optical_flow(output_folder, x, y, nframes):
    try:
        os.mkdir(output_folder)
    except OSError:
        pass
    for i in range(nframes):
        out_x = '{0}/x_{1:04d}.jpg'.format(output_folder, i+1)
        out_y = '{0}/y_{1:04d}.jpg'.format(output_folder, i+1)
        cv2.imwrite(out_x, x[i])
        cv2.imwrite(out_y, y[i])

def read_frames(v):
    img = cv2.imdecode(v, -1)
    return img

if __name__ == "__main__":
    f = FlowExtractor(dev_id=0)
    source = '/z/home/kylemin/dataset/Kinetics-new/Kinetics-400h/'

    sp = 'train'
    split = 48
    div = 8

    #sp = 'val'
    #split = 16
    #div = 2

    for i in xrange(split):
        for j in xrange(div):
            file_name = source + sp + '_flow_%d-%d.h5'%(i+1, j+1)
            if div == 1:
                file_name = source + sp + '_flow_%d.h5'%(i+1)
            time.sleep(random.randint(1,100)/100.)
            if not os.path.isfile(file_name):
                print (file_name)
                with h5py.File(file_name, 'w') as hf:
                    with h5py.File(source + sp + '_%d.h5'%(i+1), 'r') as h:
                        num_video = len(h['shapes'])
                        d = num_video//div
                        s = j*d
                        e = s+d
                        if j+1 == div:
                            e = num_video
                        rng = range(s, e)
                        videos = h['videos'][rng]
                        labels = h['labels'][rng]
                        shapes = h['shapes'][rng]

                    num_video = len(shapes)
                    dset_labels = hf.create_dataset('labels', shape=(num_video,), dtype=int)
                    dset_shapes = hf.create_dataset('shapes', shape=(num_video,4), dtype=int)
                    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                    dset_u = hf.create_dataset('u', shape=(num_video,250), dtype=dt)
                    dset_v = hf.create_dataset('v', shape=(num_video,250), dtype=dt)
                    np_u = np.ndarray((num_video, 250), dtype='object')
                    np_v = np.ndarray((num_video, 250), dtype='object')

                    print ('start decoding')
                    for k in xrange(num_video):
                        frame_list = Parallel(n_jobs=50, prefer="threads")(delayed(read_frames)(videos[k][l]) for l in range(shapes[k][0]))
                        st = time.time()
                        x, y, nframes = f.extract_flow(frame_list)
                        for l in xrange(250):
                            if l < nframes:
                                u = cv2.imencode('.jpg', x[l], [cv2.IMWRITE_JPEG_QUALITY, 99])[1]
                                v = cv2.imencode('.jpg', y[l], [cv2.IMWRITE_JPEG_QUALITY, 99])[1]
                                np_u[k, l] = np.fromstring(u.squeeze().tostring(), dtype='uint8')
                                np_v[k, l] = np.fromstring(v.squeeze().tostring(), dtype='uint8')
                            else:
                                np_u[k, l] = np.array([], dtype=np.uint8)
                                np_v[k, l] = np.array([], dtype=np.uint8)
                        shapes[k][0] = nframes
                        print ('extract_flow: %5d/%5d, %.4f' % (k+1, num_video, time.time()-st))

                    dset_u[...] = np_u
                    dset_v[...] = np_v
                    dset_shapes[...] = shapes
                    dset_labels[...] = labels
