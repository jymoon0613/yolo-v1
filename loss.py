"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S # ! 7
        self.B = B # ! 2
        self.C = C # ! 20

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        # ! bbox regression, bg의 confidence score에 대한 loss 가중치 설정
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):

        # ! predictions = (B, 1470)     -> 입력 이미지에 대한 모델 예측값
        # ! target      = (B, 7, 7, 30) -> 입력 이미지에 대한 target matrix

        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        # ! Reshape 진행
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # ! predictions = (B, 7, 7, 30)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        # ! YOLOv1은 하나의 grid cell에서 두 개의 bbox를 예측함
        # ! 이때 두 개의 bbox 중 gt_box와 IoU가 더 큰 bbox에만 gt bbox를 할당함
        # ! 이를 위해 두 개의 예측된 bbox와 gt bbox를 IoU를 모두 계산함
        # ! utils.intersection_over_union 참조
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        # ! iou_b1 = (B, 7, 7, 1) -> 모든 첫 번째 예측 bbox와 gt bbox 간의 IoU 값
        # ! iou_b2 = (B, 7, 7, 1) -> 모든 두 번째 예측 bbox와 gt bbox 간의 IoU 값

        # ! 결과 concat
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # ! (2, B, 7, 7, 1)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        # ! 두 개의 예측 bbox 중 gt bbox와 IoU가 더 큰 bbox를 식별
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # ! iou_maxes = (B, 7, 7, 1) -> 식별된 bbox와 gt bbox의 IoU 값
        # ! bestbox   = (B, 7, 7, 1) -> 식별된 bbox의 indices (0 or 1)
        # ! index가 0인 경우 첫 번째 bbox(predictions[..., 20:25]), 1인 경우 두 번째 bbox(predictions[..., 25:30])를 의미

        # ! bbox regression 수행
        # ! 먼저, 해당 grid cell에 object가 존재하는지에 대한 정보를 추출
        exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i
        # ! exists_box = (B, 7, 7, 1) -> 0은 object가 없음을 의미하고, 1은 있음을 의미함 (Iobj)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        # ! bbox regression을 위한 예측값 설정
        # ! (i) 해당 grid cell에 object가 존재하는 경우, (ii) gt bbox와의 IoU가 가장 큰 bbox에 대해서만 loss 고려
        box_predictions = exists_box * (

            (   
                # ! 두 번째 예측 bbox가 best bbox인 경우 bbox 예측값 가져오기
                bestbox * predictions[..., 26:30] 

                # ! 첫 번째 예측 bbox가 best bbox인 경우 bbox 예측값 가져오기
                + (1 - bestbox) * predictions[..., 21:25] 

            )

        ) # ! (B, 7, 7, 4)

        # ! 마찬가지로 bbox regression을 위한 target값 설정
        # ! object가 존재하는 grid cell의 gt bbox 좌표만 가져옴 (나머지는 0)
        box_targets = exists_box * target[..., 21:25] # ! (B, 7, 7, 4)

        # Take sqrt of width, height of boxes to ensure that
        # ! Paper에서 w와 h에는 sqrt를 취해준 뒤 mse 계산
        # ! 따라서 예측 bbox와 gt bbox의 w와 h에 sqrt를 취해줌
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        ) # ! (B, 7, 7, 4)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4]) # ! (B, 7, 7, 4)

        # ! bbox_reg loss 계산 = MSE loss 
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2), # ! (49B, 4)
            torch.flatten(box_targets, end_dim=-2), # ! (49B, 4)
        ) # ! -> scalar loss value

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        # ! Confidence score에 대한 loss 계산
        # ! 마찬가지로 (i) 해당 grid cell에 object가 존재하는 경우, (ii) gt bbox와의 IoU가 가장 큰 bbox에 대해서만 loss 고려
        pred_box = (

            # ! 두 번째 예측 bbox가 best bbox인 경우 confidence score 예측값 가져오기
            bestbox * predictions[..., 25:26] 

            # ! 첫 번째 예측 bbox가 best bbox인 경우 confidence score 예측값 가져오기
            + (1 - bestbox) * predictions[..., 20:21] 

        ) # ! (B, 7, 7, 1)

        # ! Confidence loss 계산 = MSE loss
        # ! exist_box를 곱해주어 object가 존재하는 grid cell에 대해서만 loss 계산
        object_loss = self.mse(

            # ! object가 존재하는 grid cell에서 예측된 confidence score
            torch.flatten(exists_box * pred_box), # ! (49B)

            # ! object가 존재하는 grid cell에서의 gt confidence score
            torch.flatten(exists_box * target[..., 20:21]), # ! (49B)

        ) # ! -> scalar loss value

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        # ! Object가 존재하지 않을 때의 confidence score에 대한 loss 계산 = MSE loss
        no_object_loss = self.mse(

            # ! Object가 존재하지 않는 grid cell에서 예측된 confidence score (첫 번째 bbox)
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1), # ! (B, 49)

            # ! Object가 존재하지 않는 grid cell에서의 gt confidence score = 0
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1), # ! (B, 49)

        ) # ! -> scalar loss value

        no_object_loss += self.mse(

            # ! Object가 존재하지 않는 grid cell에서 예측된 confidence score (두 번째 bbox)
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1), # ! (B, 49)

            # ! Object가 존재하지 않는 grid cell에서의 gt confidence score = 0
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1) # ! (B, 49)

        ) # ! -> scalar loss value

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # ! Object classification loss 계산 = MSE loss
        class_loss = self.mse(

            # ! object가 존재하는 grid cell에서 예측된 class score 
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,), # ! (49B, 20)

            # ! object가 존재하는 grid cell에서의 gt class score
            torch.flatten(exists_box * target[..., :20], end_dim=-2,), # ! (49B, 20)
        ) # ! -> scalar loss value

        # ! loss 가중치 곱해주어 최종 계산
        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        # ! loss -> scalar loss value

        return loss
