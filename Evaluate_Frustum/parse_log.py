import matplotlib.pyplot as plt

file1 = open('/Weights/default_carpedcyc_kitti_2021-08-13-22-1024pts/log_train.txt', 'r')
lines1 = file1.readlines()

boxIOU_str = "box IoU(ground/3D)"
boxAcc_str = "box estimation accuracy"
segAcc_str = "segmentation accuracy"
epoch_str = "Epoch"

ep_num = 0

boxIOU3d = []
boxIOU2d = []
boxAcc = []
segAcc = []

boxIOU3d_test = []
boxIOU2d_test = []
boxAcc_test = []
segAcc_test = []

for i, line in enumerate(lines1):
    if epoch_str in line:
        ep_num += 1
    if boxIOU_str in line:
        if "test" in line:
            sp = line.split(":")[1].split("/")
            boxIOU2d_test.append(float(sp[0]))
            boxIOU3d_test.append(float(sp[1]))
        else:
            sp = line.split(":")[1].split("/")
            boxIOU2d.append(float(sp[0]))
            boxIOU3d.append(float(sp[1]))
    if boxAcc_str in line:
        if "test" in line:
            boxAcc_test.append(float(line.split(":")[1]))
        else:
            boxAcc.append(float(line.split(":")[1]))
    if segAcc_str in line:
        if "test" in line:
            segAcc_test.append(float(line.split(":")[1]))
        else:
            segAcc.append(float(line.split(":")[1]))

plt.plot(boxIOU2d, label="boxIOU2d", color='#1f77b4')
plt.plot(boxIOU3d, label="boxIOU3d", color='#ff7f0e')
plt.plot(boxAcc, label="boxAcc", color='#2ca02c')
plt.plot(segAcc, label="segAcc", color='#d62728')
plt.plot(boxIOU2d_test, linestyle="--", color='#1f77b4')
plt.plot(boxIOU3d_test, linestyle="--", color='#ff7f0e')
plt.plot(boxAcc_test, linestyle="--", color='#2ca02c')
plt.plot(segAcc_test, linestyle="--", color='#d62728')
plt.title("Acc and IOU")
plt.legend()
plt.show()

