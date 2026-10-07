"""ctypes wrapper around bin/liblab6.so (CUDA matrix multiply and 2-D convolution).

    from lab6 import Lab6
    gpu = Lab6()
    C, kernel_ms = gpu.matmul(A, B, kernel="regblock")
    out, kernel_ms = gpu.conv2d(img_uint8, filt_float, tiled=True)
"""
import ctypes
import os
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_F32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
_U8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")


class Lab6:
    KERNELS = {"tiled16": 0, "regblock": 1, "vec4": 2, "cublas": 3}

    def __init__(self, path=os.path.join(_ROOT, "bin", "liblab6.so")):
        lib = ctypes.CDLL(path)
        lib.gpu_matrix_multiply.argtypes = [_F32, _F32, _F32, ctypes.c_int]
        lib.gpu_matrix_multiply.restype = ctypes.c_int
        lib.lab6_create.argtypes = []
        lib.lab6_create.restype = ctypes.c_void_p
        lib.lab6_destroy.argtypes = [ctypes.c_void_p]
        lib.lab6_last_kernel_ms.argtypes = [ctypes.c_void_p]
        lib.lab6_last_kernel_ms.restype = ctypes.c_float
        lib.lab6_matmul.argtypes = [ctypes.c_void_p, _F32, _F32, _F32, ctypes.c_int, ctypes.c_int]
        lib.lab6_matmul.restype = ctypes.c_int
        lib.lab6_conv2d.argtypes = [ctypes.c_void_p, _U8, _F32, ctypes.c_int, _F32, ctypes.c_int, ctypes.c_int]
        lib.lab6_conv2d.restype = ctypes.c_int
        self.lib = lib
        self.ctx = lib.lab6_create()
        if not self.ctx:
            raise RuntimeError("lab6_create failed: no usable CUDA device")

    @staticmethod
    def _check(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} failed with code {rc}")

    def matmul_handout(self, A, B):
        """Handout Step 7.3 interface: allocate, copy, compute, copy back, free."""
        N = A.shape[0]
        A = np.ascontiguousarray(A, np.float32)
        B = np.ascontiguousarray(B, np.float32)
        C = np.empty((N, N), np.float32)
        self._check(self.lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N), "gpu_matrix_multiply")
        return C

    def matmul(self, A, B, kernel="regblock"):
        """Persistent device buffers. Returns (C, kernel_ms)."""
        N = A.shape[0]
        A = np.ascontiguousarray(A, np.float32)
        B = np.ascontiguousarray(B, np.float32)
        C = np.empty((N, N), np.float32)
        self._check(self.lib.lab6_matmul(self.ctx, A.ravel(), B.ravel(), C.ravel(), N,
                                         self.KERNELS[kernel]), f"matmul({kernel})")
        return C, self.lib.lab6_last_kernel_ms(self.ctx)

    def conv2d(self, img, filt, tiled=True):
        """img: square uint8 array, filt: odd square float array. Returns (float32 image, kernel_ms)."""
        M = img.shape[0]
        N = filt.shape[0]
        img = np.ascontiguousarray(img, np.uint8)
        filt = np.ascontiguousarray(filt, np.float32)
        out = np.empty((M, M), np.float32)
        self._check(self.lib.lab6_conv2d(self.ctx, img.ravel(), out.ravel(), M, filt.ravel(), N,
                                         1 if tiled else 0), "conv2d")
        return out, self.lib.lab6_last_kernel_ms(self.ctx)

    def close(self):
        if self.ctx:
            self.lib.lab6_destroy(self.ctx)
            self.ctx = None

    def __del__(self):
        self.close()
