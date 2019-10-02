Extracting dense flow field given a video.

#### Dependencies:
- LibZip:
to install on ubuntu ```apt-get install libzip-dev``` on mac ```brew install libzip```

### Install
```
git clone --recursive http://github.com/kylemin/dense_flow
mkdir build && cd build
cmake .. && make -j

* It works only on python2 because of np.string
* In CMakeLists.txt, FIND_PACKAGE(PythonLibs 2 REQUIRED)
* It takes about 0.1s per frame
```

### Usage
```
./extract_gpu -f=test.avi -x=tmp/flow_x -y=tmp/flow_y -i=tmp/image -b=20 -t=1 -d=0 -s=1 -o=dir
```
- `test.avi`: input video
- `tmp`: folder containing RGB images and optical flow images
- `dir`: output generated images to folder. if set to `zip`, will write images to zip files instead.

* python extract_flow.py or
* python extract_flow_h.py

### Warp Flow
The warp optical flow is used in the following paper

```
@inproceedings{TSN2016ECCV,
  author    = {Limin Wang and
               Yuanjun Xiong and
               Zhe Wang and
               Yu Qiao and
               Dahua Lin and
               Xiaoou Tang and
               Luc {Van Gool}},
  title     = {Temporal Segment Networks: Towards Good Practices for Deep Action Recognition},
  booktitle   = {ECCV},
  year      = {2016},
}
```

To extract warp flow, use the command
```
./extract_warp_gpu -f test.avi -x tmp/flow_x -y tmp/flow_y -i tmp/image -b 20 -t 1 -d 0 -s 1 -o dir
```
