set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/home/linuxbrew/.linuxbrew/opt/gcc@11/bin/gcc-11")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.3.52")
set(CMAKE_CUDA_DEVICE_LINKER "/usr/local/cuda-12.3/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/usr/local/cuda-12.3/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_STANDARD_LATEST "20")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17;cuda_std_20")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "cuda_std_20")
set(CMAKE_CUDA23_COMPILE_FEATURES "")
set(CMAKE_CUDA26_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "11.5")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)
set(CMAKE_CUDA_LINKER_DEPFILE_SUPPORTED )

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/local/cuda-12.3")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/usr/local/cuda-12.3")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "12.3.52")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/local/cuda-12.3")

set(CMAKE_CUDA_ARCHITECTURES_ALL "50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "89-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda-12.3/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda-12.3/targets/x86_64-linux/lib/stubs;/usr/local/cuda-12.3/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/include/c++/11;/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/include/c++/11/x86_64-pc-linux-gnu;/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/include/c++/11/backward;/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/lib/gcc/11/gcc/x86_64-pc-linux-gnu/11/include;/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/lib/gcc/11/gcc/x86_64-pc-linux-gnu/11/include-fixed;/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/include;/home/linuxbrew/.linuxbrew/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "gcc;gcc_s;c;gcc;gcc_s")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda-12.3/targets/x86_64-linux/lib/stubs;/usr/local/cuda-12.3/targets/x86_64-linux/lib;/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/lib/gcc/11/gcc/x86_64-pc-linux-gnu/11;/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/lib/gcc/11/gcc;/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/lib/gcc/11;/home/linuxbrew/.linuxbrew/lib/gcc/11;/home/linuxbrew/.linuxbrew/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_LINKER_LINK "")
set(CMAKE_LINKER_LLD "")
set(CMAKE_CUDA_COMPILER_LINKER "/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/bin/../libexec/gcc/x86_64-pc-linux-gnu/11/collect2 -plugin /home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/bin/../libexec/gcc/x86_64-pc-linux-gnu/11/liblto_plugin.so -plugin-opt=/home/linuxbrew/.linuxbrew/Cellar/gcc@11/11.5.0/bin/../libexec/gcc/x86_64-pc-linux-gnu/11/lto-wrapper -plugin-opt=-fresolution=/tmp/ccBHBjFQ.res -plugin-opt=-pass-through=-lgcc -plugin-opt=-pass-through=-lgcc_s -plugin-opt=-pass-through=-lc -plugin-opt=-pass-through=-lgcc -plugin-opt=-pass-through=-lgcc_s --eh-frame-hdr -m elf_x86_64 -dynamic-linker /lib64/ld-linux-x86-64.so.2 --dynamic-linker /home/linuxbrew/.linuxbrew/lib/ld.so")
set(CMAKE_CUDA_COMPILER_LINKER_ID "")
set(CMAKE_CUDA_COMPILER_LINKER_VERSION )
set(CMAKE_CUDA_COMPILER_LINKER_FRONTEND_VARIANT )
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_MT "")
