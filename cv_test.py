import cv2
print("OpenCV:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
print(cv2.dnn.getAvailableBackends())
