import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

from pointnet2.data import Indoor3DSemSeg
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

FEATURE_N = 3
class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def __init__(self):
        super(PointNet2SemSegSSG, self).__init__()

        self.use_xyz = True
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[FEATURE_N, 32, 32, 64],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=self.use_xyz,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + FEATURE_N, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 2, kernel_size=1),
        )

    def forward(self, pointcloud, features):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = pointcloud.contiguous(), torch.zeros_like(pointcloud).transpose(1,2).contiguous() #features.contiguous()
        # xyz, features = self._break_up_pc(pointcloud)
        print("xyz: ", xyz.size(), "features", features if features is None else features.size())

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        print("l_features", len(l_features), l_features[0].size())

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        out = self.fc_layer(l_features[0])
        print("out", out.size())
        return out

    # def prepare_data(self):
    #     self.train_dset = Indoor3DSemSeg(self.hparams["num_points"], train=True)
    #     self.val_dset = Indoor3DSemSeg(self.hparams["num_points"], train=False)
