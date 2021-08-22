from frustum_pointnets_v1_old import FrustumPointNetv1
from frustum_pointnets_v1_old import g_type2class, g_class2type, g_type2onehotclass, g_type_mean_size, NUM_HEADING_BIN
import numpy as np
import provider_fpointnet as provider
import torch
from torch.utils.data import DataLoader
import open3d as o3d
from tqdm import tqdm

WEIGHT_PATH = "/Weights/default_carpedcyc_kitti_2021-08-13-01-512pts/acc0.644236-epoch144.pth"

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def from_prediction_to_label_format(center, angle_class, angle_res, \
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
    # ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]], dtype=object)

    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def output_to_label(heading_scores, heading_residuals, size_scores, size_residuals, center, batch_rot_angle):
    batch_hclass_pred = np.argmax(heading_scores, 1)
    heading_cls = batch_hclass_pred

    heading_res = np.array([heading_residuals[j, batch_hclass_pred[j]] \
                            for j in range(batch_hclass_pred.shape[0])])

    size_cls = np.argmax(size_scores, 1)

    size_res = np.vstack([size_residuals[j, size_cls, :] \
                          for j in range(batch_hclass_pred.shape[0])])  # (bs,3)

    return from_prediction_to_label_format(center[0],
                                                  heading_cls[0], heading_res[0],
                                                  size_cls[0], size_res[0],
                                                  batch_rot_angle)


def run_inference(model, data):
    batch_data, batch_label, batch_center, \
    batch_hclass, batch_hres, \
    batch_sclass, batch_sres, \
    batch_rot_angle, batch_one_hot_vec = data

    batch_data = batch_data.transpose(2, 1).float()  # .cuda()
    batch_label = batch_label.float()  # .cuda()
    batch_center = batch_center.float()  # .cuda()
    batch_hclass = batch_hclass.float()  # .cuda()
    batch_hres = batch_hres.float()  # .cuda()
    batch_sclass = batch_sclass.float()  # .cuda()
    batch_sres = batch_sres.float()  # .cuda()
    batch_rot_angle = batch_rot_angle.float()  # .cuda()
    batch_one_hot_vec = batch_one_hot_vec.float()  # .cuda()

    """
    Model
    - input: 
        first_arg (points): (1,4,n) => (x, y, z, intensity) ; n => number of points
        second_arg (classes): (1, 3) => class in one hot encoded format
    """

    model = model.eval()
    logits, mask, stage1_center, center_boxnet, \
    heading_scores, heading_residuals_normalized, heading_residuals, \
    size_scores, size_residuals_normalized, size_residuals, center = \
        model(batch_data, batch_one_hot_vec)

    center = center.detach().numpy()
    heading_scores = heading_scores.detach().numpy()
    heading_residuals = heading_residuals.detach().numpy()
    size_scores = size_scores.detach().numpy()
    size_residuals = size_residuals.detach().numpy()
    batch_center = batch_center.detach().numpy()
    batch_hclass = batch_hclass.detach().numpy()
    batch_hres = batch_hres.detach().numpy()
    batch_sclass = batch_sclass.detach().numpy()
    batch_sres = batch_sres.detach().numpy()
    batch_rot_angle = batch_rot_angle.detach().numpy()

    iou2ds, iou3ds = provider.compute_box3d_iou(
        center,
        heading_scores,
        heading_residuals,
        size_scores,
        size_residuals,
        batch_center,
        batch_hclass,
        batch_hres,
        batch_sclass,
        batch_sres)
    test_iou2d = np.sum(iou2ds)
    test_iou3d = np.sum(iou3ds)
    test_iou3d_acc = np.sum(iou3ds >= 0.7)

    ph, pw, pl, ptx, pty, ptz, pry = output_to_label(heading_scores, heading_residuals, size_scores, size_residuals, center, batch_rot_angle)

    h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(batch_center[0],
                                                              batch_hclass[0], batch_hres[0],
                                                              batch_sclass[0], batch_sres[0],
                                                              batch_rot_angle)

    # print(ph, pw, pl, ptx, pty, ptz, pry)
    # print(h, w, l, tx, ty, tz, ry)
    print("IOU2D: {:.04f}, IOU3D: {:.04f}, ACC: {:.04f}".format(test_iou2d, test_iou3d, test_iou3d_acc))

    pred_box_points = get_3d_box((pl, pw, ph), pry, (ptx, pty, ptz))
    box_points = get_3d_box((l, w, h), ry, (tx, ty, tz))
    return pred_box_points, box_points, (test_iou2d, test_iou3d, test_iou3d_acc)

NUM_POINT = 1024
BATCH_SIZE = 1

if __name__ == '__main__':
    # TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
    #     rotate_to_center=True, one_hot=True,
    #     overwritten_data_path='kitti/frustum_'+'caronly'+'_'+'val'+'.pickle')  # or carpedcyc
    TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train',
        rotate_to_center=True, one_hot=True,
        overwritten_data_path='/mnt/wato-drive/KITTI/pickle/frustum_'+'carpedcyc'+'_'+'train'+'.pickle')  # or carpedcyc
    train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=False, \
                                 num_workers=8, pin_memory=True)

    tot_iou2d = []
    tot_iou3d = []
    tot_iouacc = []

    for i, data in enumerate(train_dataloader):
        # inference
        model = FrustumPointNetv1(n_classes=3, n_channel=4)
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))['model_state_dict'])
        pred_box_points, box_points, acc = run_inference(model, data)

        batch_data, _, _, _, _, _, _, rot_angle, _ = data

        # point cloud
        batch_data = batch_data.transpose(2, 1).float().detach().numpy()
        rot_angle = rot_angle.detach().numpy()
        xyz = np.swapaxes(batch_data[0, :3, :], 0, 1)
        xyz = [rotate_pc_along_y(np.expand_dims([pt[0], pt[1], pt[2]], 0), -rot_angle).squeeze() for i, pt in enumerate(xyz)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        tot_iou2d.append(acc[0])
        tot_iou3d.append(acc[1])
        tot_iouacc.append(acc[2])
        if acc[0] > 0.5:
            # 3d bounding box
            lines = [[0, 1], [1, 2], [1, 5], [0, 3], [0, 4], [2, 3], [2, 6],
                     [3, 7], [4, 5], [5, 6], [4, 7], [6, 7]]
            colors = [[0, 1, 0] for _ in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            colors = [[1, 0, 0] for _ in range(len(lines))]
            pred_line_set = o3d.geometry.LineSet()
            pred_line_set.points = o3d.utility.Vector3dVector(pred_box_points)
            pred_line_set.lines = o3d.utility.Vector2iVector(lines)
            pred_line_set.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd, line_set, pred_line_set])
            input()
        if i % 1000 == 0:
            print(i, np.mean(tot_iou2d), np.mean(tot_iou3d), np.mean(tot_iouacc))