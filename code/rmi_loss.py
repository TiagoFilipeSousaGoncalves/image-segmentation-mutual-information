# Source: https://github.com/ZJULearning/RMI

# Imports
# Python 2.X, 3.X Compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# System Imports
import math
import numpy as np

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project Imports
import rmi_loss_utils as rmi_utils



# Class: Region Mutual Information Loss
class RMILoss(nn.Module):
	
    """
	Region Mutual Information
	I(A, B) = H(A) + H(B) - H(A, B)
	This version need a lot of memory if do not downsample.
	"""
	
    # Function: __init__
    def __init__(self, num_classes, rmi_radius=3, rmi_pool_way=0, rmi_pool_size=3, rmi_pool_stride=3, loss_weight_lambda=0.5, lambda_way=1):
        super(RMILoss, self).__init__()
        
        # Initialise variables
        # Euler Number (2.718281828)
        self._euler_num = math.e
        # print(self._euler_num)

        # Pi (3.14159265)
        self._pi = math.pi
        # print(self._pi)
        
        # ln(2 * pi) (1.837877)
        self._ln_2_pi = np.log(2 * self._pi)
        # print(self._ln_2_pi)

        # Min clip value after softmax or sigmoid operations
        self._CLIP_MIN = 1e-6

        # Max clip value after softmax or sigmoid operations
        self._CLIP_MAX = 1.0

        # Add this factor to ensure the AA^T is positive definite
        self._POS_ALPHA = 5e-4						
        
        # Sum the loss per channel
        self._IS_SUM = 1



        # Number of Classes
        self.num_classes = num_classes
        
        # Radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Radius value not compatible with loss."
        self.rmi_radius = rmi_radius
        
        # Pool way
        assert rmi_pool_way in [0, 1, 2, 3], "Pool way not compatible with loss"
        self.rmi_pool_way = rmi_pool_way

        # Set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride, "Pool size and pool stride must have the same value"
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        
        # Weightt lambda
        self.weight_lambda = loss_weight_lambda

        # Lambda way
        self.lambda_way = lambda_way

        # Dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        
        # Ignore class
        self.ignore_index = 255

        return


    # Function: forward method
    def forward(self, logits_4D, labels_4D):
        
        loss = self.forward_sigmoid(logits_4D, labels_4D)
        
        return loss



    # Function: forward_sigmoid implemented by the authors (used by "forward")
    def forward_sigmoid(self, logits_4D, labels_4D):
		
        """
        Using the sigmiod operation both.
        Args:
            logits_4D 	:	[N, C, H, W], dtype=float32
            labels_4D 	:	[N, H, W], dtype=long
        """

        # label mask -- [N, H, W, 1]
        label_mask_3D = labels_4D < self.num_classes

        # valid label
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(), num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = valid_onehot_labels_4D.view([-1, self.num_classes]).requires_grad_(False)
        logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(logits_flat, target=valid_onehot_label_flat, weight=label_mask_flat.unsqueeze(dim=1), reduction='sum')
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)
        # print(f"BCE Loss: {bce_loss}")

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + self._CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)
        # print(f"RMI Loss: {rmi_loss}")

        # add together
        final_loss = (self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way else bce_loss + rmi_loss * self.weight_lambda)
        # print(f"Final Loss: {final_loss}")

        return final_loss


    # Function: rmi_lower_bound implemented by the authors
    def rmi_lower_bound(self, labels_4D, probs_4D):

        """
        calculate the lower bound of the region mutual information.
        Args:
            labels_4D 	:	[N, C, H, W], dtype=float32
            probs_4D 	:	[N, C, H, W], dtype=float32
        """

        assert labels_4D.size() == probs_4D.size(), "labels_4D.size() must be equal to probs_4D.size()"

        p, s = self.rmi_pool_size, self.rmi_pool_stride

        if self.rmi_pool_stride > 1:
            
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")

        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = rmi_utils.map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))

        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * self._POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * rmi_utils.log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * self._POS_ALPHA)
        # rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        # is_half = False
        # if is_half:
        #	rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(rmi_per_class) if self._IS_SUM else torch.mean(rmi_per_class)

        return rmi_loss



# Test
# _ = RMILoss(num_classes=21)