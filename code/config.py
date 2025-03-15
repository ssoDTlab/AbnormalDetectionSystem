import threading
TF_ENABLE_ONEDNN_OPTS = 0

sema0_1 = threading.Semaphore(0)
sema0_2 = threading.Semaphore(0)
sema1 = threading.Semaphore(0)
sema2 = threading.Semaphore(0)

sema3 = threading.Semaphore(1) 


Processing_stop = False
