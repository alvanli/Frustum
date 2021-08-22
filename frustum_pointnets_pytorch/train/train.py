import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util import FrustumPointNetLoss
import argparse
import importlib
import time
import ipdb
import numpy as np
import random
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import provider
from configs.config import cfg
from configs.config import merge_cfg_from_file
from configs.config import merge_cfg_from_list
from configs.config import assert_and_infer_cfg
from utils import import_from_file

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='cfgs/fpointnet/fpointnet_v1_kitti.yaml', help='Config file for training (and optionally testing)')
parser.add_argument('opts',help='See configs/config.py for all options',default=None,nargs=argparse.REMAINDER)
parser.add_argument('--debug', default=False, action='store_true',help='debug mode')

args = parser.parse_args()
if args.cfg is not None:
    merge_cfg_from_file(args.cfg)

if args.opts is not None:
    merge_cfg_from_list(args.opts)

assert_and_infer_cfg()

if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

# Set configurations
CONFIG_FILE = args.cfg
RESUME = cfg.RESUME
OUTPUT_DIR = cfg.OUTPUT_DIR
USE_TFBOARD = cfg.USE_TFBOARD
NUM_WORKERS = cfg.NUM_WORKERS
FROM_RGB_DET = cfg.FROM_RGB_DET
## TRAIN
TRAIN_FILE = cfg.TRAIN.FILE
BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
START_EPOCH = cfg.TRAIN.START_EPOCH
MAX_EPOCH = cfg.TRAIN.MAX_EPOCH
OPTIMIZER = cfg.TRAIN.OPTIMIZER
BASE_LR = cfg.TRAIN.BASE_LR
MIN_LR = cfg.TRAIN.MIN_LR
GAMMA = cfg.TRAIN.GAMMA
LR_STEPS = cfg.TRAIN.LR_STEPS
MOMENTUM = cfg.TRAIN.MOMENTUM
WEIGHT_DECAY = cfg.TRAIN.WEIGHT_DECAY
NUM_POINT = cfg.TRAIN.NUM_POINT
TRAIN_SETS = cfg.TRAIN.TRAIN_SETS
## TEST
TEST_FILE = cfg.TEST.FILE
TEST_BATCH_SIZE = cfg.TEST.BATCH_SIZE
TEST_NUM_POINT = cfg.TEST.NUM_POINT
TEST_SETS = cfg.TEST.TEST_SETS
## MODEL
MODEL_FILE = cfg.MODEL.FILE
NUM_CLASSES = cfg.MODEL.NUM_CLASSES
## DATA
DATA_FILE = cfg.DATA.FILE
DATASET = cfg.DATA.DATASET
DATAROOT = cfg.DATA.DATA_ROOT
OBJTYPE = cfg.DATA.OBJTYPE
SENSOR = cfg.DATA.SENSOR
ROTATE_TO_CENTER = cfg.DATA.ROTATE_TO_CENTER
NUM_CHANNEL = cfg.DATA.NUM_CHANNEL
NUM_SAMPLES = cfg.DATA.NUM_SAMPLES

strtime = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))
strtime = strtime[4:8]
NAME = '_'.join(OUTPUT_DIR.split('/')) + '_' + strtime
print(NAME)
MODEL = import_from_file(MODEL_FILE) # import network module
LOG_DIR = OUTPUT_DIR + '/' + NAME
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (CONFIG_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')



# Load Frustum Datasets.
if 'frustum_pointnet' in MODEL_FILE:
    gen_ref = False
elif 'frustum_convnet' in MODEL_FILE:
    gen_ref = True
else:
    print("Wrong model parameter.")
    exit(0)

if 'fusion' in MODEL_FILE:
    with_image = True
else:
    with_image = False

provider = import_from_file(DATA_FILE)

TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split=TRAIN_SETS,
        rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True,
        overwritten_data_path=TRAIN_FILE,
        gen_ref = gen_ref, with_image = with_image)
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split=TEST_SETS,
        rotate_to_center=True, one_hot=True,
        overwritten_data_path=TEST_FILE,
        gen_ref = gen_ref, with_image = with_image)
train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS,pin_memory=True)
test_dataloader = DataLoader(TEST_DATASET, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS,pin_memory=True)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def test_one_epoch(model, loader):
    time1 = time.perf_counter()

    test_losses = {
        'total_loss': 0.0,
        'cls_loss': 0.0,  # fconvnet
        'mask_loss': 0.0,  # fpointnet
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }
    test_metrics = {
        'seg_acc': 0.0,  # fpointnet
        'cls_acc': 0.0,  # fconvnet
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.7': 0.0,
    }

    n_batches = 0
    for i, data_dicts in tqdm(enumerate(loader), \
                              total=len(loader), smoothing=0.9):
        n_batches += 1
        # for debug
        if args.debug == True:
            if i == 1:
                break

        data_dicts_var = {key: value for key, value in data_dicts.items()}

        model = model.eval()

        with torch.no_grad():
            losses, metrics = model(data_dicts_var)

        for key in test_losses.keys():
            if key in losses.keys():
                test_losses[key] += losses[key].detach().item()
        for key in test_metrics.keys():
            if key in metrics.keys():
                test_metrics[key] += metrics[key]

    for key in test_losses.keys():
        test_losses[key] /= n_batches
    for key in test_metrics.keys():
        test_metrics[key] /= n_batches

    time2 = time.perf_counter()
    print('test time:%.2f s/batch'%((time2-time1)/n_batches))
    return test_losses, test_metrics

def train():
    ''' Main function for training and simple evaluation. '''
    start= time.perf_counter()
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    blue = lambda x: '\033[94m' + x + '\033[0m'

    # set model
    if 'frustum_pointnets_v1' in MODEL_FILE:
        from models.frustum_pointnets_v1 import FrustumPointNetv1
        model = FrustumPointNetv1(n_classes=NUM_CLASSES,n_channel=NUM_CHANNEL) #.cuda()

    # set optimizer and scheduler
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=BASE_LR,
            betas=(0.9, 0.999),eps=1e-08,
            weight_decay=WEIGHT_DECAY)
    '''
    def lr_func(epoch, init=BASE_LR, step_size=LR_STEPS, gamma=GAMMA, eta_min=MIN_LR):
        f = gamma**(epoch//LR_STEPS)
        if init*f>eta_min:
            return f
        else:
            return 0.01#0.001*0.01 = eta_min
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)
    '''
    if len(LR_STEPS) > 1:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEPS, gamma=GAMMA)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEPS[0], gamma=GAMMA)
    # train
    if USE_TFBOARD:
        if os.path.exists('runs/' + NAME):
            print('name has been existed')
            shutil.rmtree('runs/' + NAME)
        writer = SummaryWriter('runs/' + NAME)

    num_batch = len(TRAIN_DATASET) / BATCH_SIZE
    best_iou3d_70 = 0.0
    best_epoch = 1
    best_file = ''

    for epoch in range(MAX_EPOCH):
        log_string('**** cfg:%s ****' % (args.cfg))
        log_string('**** output_dir:%s ****' % (OUTPUT_DIR))
        log_string('**** EPOCH %03d ****' % (epoch + 1))
        sys.stdout.flush()
        log_string('Epoch %d/%s:' % (epoch + 1, MAX_EPOCH))
        # record for one epoch
        train_total_loss = 0.0
        train_iou2d = 0.0
        train_iou3d = 0.0
        train_acc = 0.0
        train_iou3d_70 = 0.0

        train_losses = {
            'total_loss': 0.0,
            'cls_loss': 0.0, #fconvnet
            'mask_loss': 0.0,#fpointnet
            'heading_class_loss': 0.0,
            'size_class_loss': 0.0,
            'heading_residual_normalized_loss': 0.0,
            'size_residual_normalized_loss': 0.0,
            'stage1_center_loss': 0.0,
            'corners_loss': 0.0
        }
        train_metrics = {
            'seg_acc': 0.0,#fpointnet
            'cls_acc': 0.0,#fconvnet
            'iou2d': 0.0,
            'iou3d': 0.0,
            'iou3d_0.7': 0.0,
        }
        n_batches = 0
        for i, data_dicts in tqdm(enumerate(train_dataloader),\
                total=len(train_dataloader), smoothing=0.9):
            n_batches += 1
            #for debug
            if args.debug==True:
                if i==1 :
                    break
            '''
            data after frustum rotation
            1. For Seg
            batch_data:[32, 2048, 4], pts in frustum, 
            batch_label:[32, 2048], pts ins seg label in frustum,
            2. For T-Net
            batch_center:[32, 3],
            3. For Box Est.
            batch_hclass:[32],
            batch_hres:[32],
            batch_sclass:[32],
            batch_sres:[32,3],
            4. Others
            batch_rot_angle:[32],alpha, not rotation_y,
            batch_one_hot_vec:[32,3],
            '''
            data_dicts_var = {key: value for key, value in data_dicts.items()}

            optimizer.zero_grad()
            model = model.train()

            losses, metrics = model(data_dicts_var)

            total_loss = losses['total_loss']
            total_loss.backward()

            optimizer.step()

            for key in train_losses.keys():
                if key in losses.keys():
                    train_losses[key] += losses[key].detach().item()
            for key in train_metrics.keys():
                if key in metrics.keys():
                    train_metrics[key] += metrics[key]

        for key in train_losses.keys():
            train_losses[key] /= n_batches
        for key in train_metrics.keys():
            train_metrics[key] /= n_batches

        log_string('[%d: %d/%d] train' % (epoch + 1, i, len(train_dataloader)))
        for key, value in train_losses.items():
            if value < 1e-6: continue
            log_string(str(key)+':'+"%.6f"%(value))
        for key, value in train_metrics.items():
            if value < 1e-6: continue
            log_string(str(key)+':'+"%.6f"%(value))

        # test one epoch
        test_losses, test_metrics = test_one_epoch(model,test_dataloader)
        log_string('[%d: %d/%d] %s' % (epoch + 1, i, len(train_dataloader),blue('test')))
        for key, value in test_losses.items():
            if value < 1e-6: continue
            log_string(str(key)+':'+"%.6f"%(value))
        for key, value in test_metrics.items():
            if value < 1e-6: continue
            log_string(str(key)+':'+"%.6f"%(value))

        scheduler.step()
        if MIN_LR > 0:
            if scheduler.get_lr()[0] < MIN_LR:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = MIN_LR
        log_string("learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))

        if USE_TFBOARD:
            writer.add_scalar('train_total_loss',train_losses['total_loss'],epoch)
            writer.add_scalar('train_iou3d_0.7',train_metrics['iou3d_0.7'],epoch)
            writer.add_scalar('test_total_loss',test_losses['total_loss'],epoch)
            writer.add_scalar('test_iou3d_0.7',test_metrics['iou3d_0.7'],epoch)

        if test_metrics['iou3d_0.7'] >= best_iou3d_70:
            best_iou3d_70 = test_metrics['iou3d_0.7']
            best_epoch = epoch + 1
            if epoch > MAX_EPOCH / 5:
                savepath = LOG_DIR + '/acc%.3f-epoch%03d.pth' % \
                           (test_metrics['iou3d_0.7'], epoch)
                log_string('save to:'+str(savepath))
                if os.path.exists(best_file):
                    os.remove(best_file)# update to newest best epoch
                best_file = savepath
                state = {
                    'epoch': epoch + 1,
                    'train_iou3d_0.7': train_metrics['iou3d_0.7'],
                    'test_iou3d_0.7': test_metrics['iou3d_0.7'],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state,savepath)
                log_string('Saving model to %s'%savepath)
        log_string('Best Test acc: %f(Epoch %d)' % (best_iou3d_70, best_epoch))
    log_string("Time {} hours".format(float(time.perf_counter()-start)/3600))
    if USE_TFBOARD:
        writer.close()

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    print('Your args:')
    print(args)
    train()
    LOG_FOUT.close()


