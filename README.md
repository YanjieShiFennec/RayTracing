https://github.com/RayTracing/raytracing.github.io

https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

Prerequisite:
1. Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Download Visual Studio 2022: https://visualstudio.microsoft.com/zh-hans/downloads/ ,
install MSVC c143 - VS 2022 C++ x64/x86 生成工具 and Windows 11 SDK
3. Clion 环境配置 CMake 和 Toolchains:
Build, Execution, Deployment
Toolchains: new Visual Studio, Toolset: D:\Program Files\Microsoft Visual Studio\2022\Community
CMake: CMake options: -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe"