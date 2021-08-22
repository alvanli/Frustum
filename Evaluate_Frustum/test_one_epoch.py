import numpy as np
import torch
import provider_fpointnet as provider

def test_one_epoch(model, loader):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    seg_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []

    test_n_samples = 0
    test_iou2d = 0.0
    test_iou3d = 0.0
    test_acc = 0.0
    test_iou3d_acc = 0.0

    for i, data in tqdm(enumerate(loader), \
                        total=len(loader), smoothing=0.9):
        test_n_samples += data[0].shape[0]
        '''
        batch_data:[32, 2048, 4], pts in frustum
        batch_label:[32, 2048], pts ins seg label in frustum
        batch_center:[32, 3],
        batch_hclass:[32],
        batch_hres:[32],
        batch_sclass:[32],
        batch_sres:[32,3],
        batch_rot_angle:[32],
        batch_one_hot_vec:[32,3],
        '''
        # 1. Load data
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = data

        batch_data = batch_data.transpose(2, 1).float()#.cuda()
        batch_label = batch_label.float()#.cuda()
        batch_center = batch_center.float()#.cuda()
        batch_hclass = batch_hclass.float()#.cuda()
        batch_hres = batch_hres.float()#.cuda()
        batch_sclass = batch_sclass.float()#.cuda()
        batch_sres = batch_sres.float()#.cuda()
        batch_rot_angle = batch_rot_angle.float()#.cuda()
        batch_one_hot_vec = batch_one_hot_vec.float()#.cuda()

        # 2. Eval one batch
        model = model.eval()
        logits, mask, stage1_center, center_boxnet, \
        heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals, center = \
            model(batch_data, batch_one_hot_vec)
        # logits:[32, 1024, 2] , mask:[32, 1024]

        # 4. compute seg acc, IoU and acc(IoU)
        correct = torch.argmax(logits, 2).eq(batch_label.detach().long()).cpu().numpy()
        accuracy = np.sum(correct) / float(NUM_POINT)
        test_acc += accuracy

        logits = logits.cpu().detach().numpy()
        center = center.cpu().detach().numpy()
        heading_scores = heading_scores.cpu().detach().numpy()
        heading_residuals = heading_residuals.cpu().detach().numpy()
        size_scores = size_scores.cpu().detach().numpy()
        size_residuals = size_residuals.cpu().detach().numpy()
        batch_rot_angle = batch_rot_angle.cpu().detach().numpy()
        batch_center = batch_center.cpu().detach().numpy()
        batch_hclass = batch_hclass.cpu().detach().numpy()
        batch_hres = batch_hres.cpu().detach().numpy()
        batch_sclass = batch_sclass.cpu().detach().numpy()
        batch_sres = batch_sres.cpu().detach().numpy()

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
        test_iou2d += np.sum(iou2ds)
        test_iou3d += np.sum(iou3ds)
        test_iou3d_acc += np.sum(iou3ds >= 0.7)

        # 5. Compute and write all Results
        batch_output = np.argmax(logits, 2)  # mask#torch.Size([32, 1024])
        batch_center_pred = center  # _boxnet#torch.Size([32, 3])
        batch_hclass_pred = np.argmax(heading_scores, 1)  # (32,)
        batch_hres_pred = np.array([heading_residuals[j, batch_hclass_pred[j]] \
                                    for j in range(batch_data.shape[0])])  # (32,)
        # batch_size_cls,batch_size_res
        batch_sclass_pred = np.argmax(size_scores, 1)  # (32,)
        batch_sres_pred = np.vstack([size_residuals[j, batch_sclass_pred[j], :] \
                                     for j in range(batch_data.shape[0])])  # (32,3)

        # batch_scores
        batch_seg_prob = softmax(logits)[:, :, 1]  # (32, 1024, 2) ->(32, 1024)
        batch_seg_mask = np.argmax(logits, 2)  # BxN

        mask_max_prob = np.max(batch_seg_prob * batch_seg_mask, 1)

        for j in range(batch_output.shape[0]):
            ps_list.append(batch_data[j, ...])
            seg_list.append(batch_label[j, ...])
            segp_list.append(batch_output[j, ...])
            center_list.append(batch_center_pred[j, :])
            heading_cls_list.append(batch_hclass_pred[j])
            heading_res_list.append(batch_hres_pred[j])
            size_cls_list.append(batch_sclass_pred[j])
            size_res_list.append(batch_sres_pred[j, :])
            rot_angle_list.append(batch_rot_angle[j])

    return test_iou2d / test_n_samples, \
           test_iou3d / test_n_samples, \
           test_acc / test_n_samples, \
           test_iou3d_acc / test_n_samples, \


