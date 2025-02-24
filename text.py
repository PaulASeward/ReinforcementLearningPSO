import ctypes
try:
    cudnn = ctypes.CDLL("libcudnn.so")
    get_version = cudnn.cudnnGetVersion
    get_version.restype = ctypes.c_size_t
    version = get_version()
    print("Loaded cuDNN version (as an integer):", version)
    # For cuDNN 8.9.4, you’d expect something like: 8094 (i.e., 8*1000 + 9*100 + 4)
except Exception as e:
    print("Error loading cuDNN:", e)