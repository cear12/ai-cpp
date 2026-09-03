# ai-cpp

Two small, independent C++17 AI/CV projects: a from-scratch feed-forward
neural network trained on XOR (zero external dependencies), and an OpenCV
computer-vision demo (face/eye detection, multi-object tracking, optical
flow, edge detection).

## neural-network/

A feed-forward network (configurable topology, sigmoid hidden layers,
linear output layer, manual backpropagation) with no dependencies beyond
the C++ standard library -- builds and runs anywhere a C++17 compiler does.

```bash
cmake -S neural-network -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/neural_network
```

Trains on XOR (10,000 epochs) and prints a PASS/FAIL check for each of the
4 input combinations, plus a round-trip check of `SaveModel()`/`LoadModel()`.
**Verified locally**: converges to near-zero error and predicts XOR
correctly (outputs within 0.1 of the 0/1 target) every run.

A couple of real issues were fixed along the way: `SaveModel()` was newly
implemented (it was declared in the header but never defined anywhere), and
its first version wrote weights with `ostream`'s default 6-digit precision,
which silently truncated every saved value -- `LoadModel()` would "succeed"
but reconstruct a nearby, not identical, network. Fixed with
`std::setprecision(17)` (`std::numeric_limits<double>::max_digits10`, the
number of decimal digits required to round-trip a `double` exactly), and
there's now a test that actually compares pre- and post-round-trip
predictions bit-for-bit rather than just checking that loading didn't
crash. `Train()` also used to call `FeedForward()` a second, redundant time
per sample per epoch just to compute the epoch's error, since
`BackPropagate()` already computes it internally; `BackPropagate()` now
returns that output directly instead.

## opencv-vision/

- `FaceDetector`: Haar-cascade face + eye detection (`cv::CascadeClassifier`),
  webcam or video file.
- `ObjectTracker`: multi-object tracking (`cv::TrackerKCF`) and Lucas-Kanade
  optical flow (`cv::calcOpticalFlowPyrLK`).
- `edgeDetectionDemo()` in `main.cpp`: Canny edge detection on a live camera
  feed, shown side-by-side with the original.

**Not compiled locally** -- this sandbox has no OpenCV installation and no
root access to add one. CI installs `libopencv-dev` + `libopencv-contrib-dev`
(the `tracking` header `ObjectTracker` uses moved into OpenCV's main
modules in 4.5.1+, but pulling in contrib too maximizes the chance this
builds regardless of exactly which OpenCV version a given Ubuntu ships) and
builds it there.

**Haar cascade data files are not vendored in this repo.** `FaceDetector`'s
constructor defaults to `data/haarcascade_frontalface_alt.xml` and
`data/haarcascade_eye_tree_eyeglasses.xml`, but no `data/` directory ever
existed anywhere in this repo's history -- the original `CMakeLists.txt`
unconditionally ran `file(COPY data/ ...)`, which fails the CMake configure
step outright when the source doesn't exist, so **this project could never
have been configured, let alone built, in the state it was in.** Fixed by
making that copy step conditional on `data/` actually existing. To run this
for real, either pass your own cascade paths to `FaceDetector`'s
constructor, or populate `opencv-vision/data/` yourself -- every OpenCV
install ships these same files (Ubuntu: `apt install opencv-data`, then
look under `/usr/share/opencv4/haarcascades/`; they're also downloadable
from the main `opencv/opencv` GitHub repo's `data/haarcascades/` directory).
CI does this automatically before building.

No display/camera in CI, so the interactive demos (`processVideo()`,
`runInteractiveTracking()`, `edgeDetectionDemo()`) are not exercised
end-to-end there -- CI verifies the build succeeds, not the live video
pipeline. `FaceDetector`/`ObjectTracker`'s non-interactive methods
(`detectFaces`, `detectEyes`, `addTracker`, `updateTrackers`, the optical
flow helpers) don't fundamentally need a display to test against a static
image or a couple of synthetic frames, if you want to extend this with a
non-interactive smoke test later.

## Build requirements

- `neural-network/`: any C++17 compiler, CMake >= 3.10. No other
  dependencies.
- `opencv-vision/`: OpenCV 4.x (including the `tracking`/`video` module),
  CMake >= 3.10.
