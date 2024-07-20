import torch
import torch.nn as nn
from torch.nn import functional as F
import geomloss

class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''

    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.mfaud = nn.Sequential(nn.Conv2d(in_c, in_c // 2,  kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                  nn.InstanceNorm2d(in_c // 2 ),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c // 2, out_c, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )

        self.mfalr = nn.Sequential(nn.Conv2d(in_c, in_c // 2,  kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                                  nn.InstanceNorm2d(in_c // 2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c // 2, out_c, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )
        self.num_iterations = 5
        self.alpha = 0.1

    def forward(self, anomaly_map):
        refined_map = anomaly_map.clone()  # Clone the input anomaly map
        batch_size, channels, height, width = refined_map.size()

        for i in range(self.num_iterations):
            for direction in ['d', 'u']:
                idx = torch.arange(height)
                if direction == 'd':
                    idx = (idx + height // 2 ** (self.num_iterations - i)) % height
                elif direction == 'u':
                    idx = (idx - height // 2 ** (self.num_iterations - i)) % height
                refined_map.add_(self.alpha * F.relu(self.mfaud(refined_map[..., idx, :])))

            for direction in ['r', 'l']:
                idx = torch.arange(width)
                if direction == 'r':
                    idx = (idx + width // 2 ** (self.num_iterations - i)) % width
                elif direction == 'l':
                    idx = (idx - width // 2 ** (self.num_iterations - i)) % width
                refined_map.add_(self.alpha * F.relu(self.mfalr(refined_map[..., idx])))

        return refined_map


class MFA(nn.Module):
    def __init__(self, base=64):
        super(MFA, self).__init__()
        self.MFA1 = ProjLayer(base * 4, base * 4)
        self.MFA2 = ProjLayer(base * 8, base * 8)
        self.MFA3 = ProjLayer(base * 16, base * 16)

    def forward(self, features, features_noise=False):
        if features_noise is not False:
            return ([self.MFA1(features_noise[0]), self.MFA2(features_noise[1]), self.MFA3(features_noise[2])], \
                    [self.MFA1(features[0]), self.MFA2(features[1]), self.MFA3(features[2])])
        else:
            return [self.MFA1(features[0]), self.MFA2(features[1]), self.MFA3(features[2])]

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss


class CosineReconstruct(nn.Module):
    def __init__(self):
        super(CosineReconstruct, self).__init__()

    def forward(self, x, y):
        return torch.mean(1 - torch.nn.CosineSimilarity()(x, y))


class Revisit_RDLoss(nn.Module):
    """
    receive multiple inputs feature
    return multi-task loss:  SSOT loss, Reconstruct Loss, Contrast Loss
    """

    def __init__(self, consistent_shuffle=True):
        super(Revisit_RDLoss, self).__init__()
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05, \
                                             reach=None, diameter=10000000, scaling=0.95, \
                                             truncate=10, cost=None, kernel=None, cluster_scale=None, \
                                             debias=True, potentials=False, verbose=False, backend='auto')
        self.reconstruct = CosineReconstruct()
        self.contrast = torch.nn.CosineEmbeddingLoss(margin=0.5)

    def forward(self, noised_feature, mfa_noised_feature, mfa_normal_feature):
        """
        noised_feature : output of encoder at each_blocks : [noised_feature_block1, noised_feature_block2, noised_feature_block3]
        projected_noised_feature: list of the projection layer's output on noised_features, projected_noised_feature = projection(noised_feature)
        projected_normal_feature: list of the projection layer's output on normal_features, projected_normal_feature = projection(normal_feature)
        """

        normal_proj1 = mfa_normal_feature[0]
        normal_proj2 = mfa_normal_feature[1]
        normal_proj3 = mfa_normal_feature[2]

        abnormal_proj1, abnormal_proj2, abnormal_proj3 = mfa_noised_feature
        loss_reconstruct = self.reconstruct(abnormal_proj1, normal_proj1) + \
                           self.reconstruct(abnormal_proj2, normal_proj2) + \
                           self.reconstruct(abnormal_proj3, normal_proj3)


        return  0.1 * loss_reconstruct
