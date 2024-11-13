from abstract.Device import Device
from abstract.mutils import *
devices = []
for did in range(DEVICE_NUM):
    device = Device(device_id = did)
    devices.append(device)

while GLOBAL_TIME <= 10:
    for device in devices:
        device.execute_workload()
    GLOBAL_TIME+=1