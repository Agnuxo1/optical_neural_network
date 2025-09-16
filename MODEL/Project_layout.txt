optical_digit/
├─ CMakeLists.txt
├─ README_RUN.txt
├─ src/
│  ├─ main.cpp
│  ├─ csv_loader.hpp
│  ├─ csv_loader.cpp
│  ├─ optical_model.hpp
│  ├─ optical_model.cu
│  ├─ training.hpp
│  ├─ training.cpp
│  ├─ utils.hpp
│  ├─ utils.cpp
│  └─ kernels.cu
└─ data/
   ├─ train.csv   (label + 784 pixels)
   └─ test.csv    (784 pixels)


Build & Run (English)

Requirements

Linux (recommended) or Windows

CUDA Toolkit 11.4+ (tested on Ampere)

A recent CMake (≥ 3.18)

Configure & build

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j


Run training (100 epochs) and create submission

./optical_digit --train ../data/train.csv --test ../data/test.csv \
  --epochs 100 --batch 512 --lr 0.003 --submission submission.csv


Tip: Increase --batch if VRAM allows, e.g., 1024. For a 3090, 512–1024 works well.
