import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # ! boxes_preds  = (B, 7, 7, 4) -> 각 grid cell에서의 예측된 bbox 좌표
    # ! boxes_labels = (B, 7, 7, 4) -> 각 grid cell에서의 gt bbox 좌표

    if box_format == "midpoint": # ! True (중심좌표 (x, y) 및 너비, 높이 (w, h)로 구성됨)
        # ! (x, y, w, h)를 (x1, y1, x2, y2) 형식으로 변환
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2   # ! (B, 7, 7, 1)
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2   # ! (B, 7, 7, 1)
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2   # ! (B, 7, 7, 1)
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2   # ! (B, 7, 7, 1)
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2 # ! (B, 7, 7, 1)
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2 # ! (B, 7, 7, 1)
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2 # ! (B, 7, 7, 1)
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2 # ! (B, 7, 7, 1)

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # ! Intersection 계산으로 위해 좌표 계산
    # ! Intersection 영역의 (x1, y1, x2, y2) 계산
    x1 = torch.max(box1_x1, box2_x1) # ! (B, 7, 7, 1)
    y1 = torch.max(box1_y1, box2_y1) # ! (B, 7, 7, 1)
    x2 = torch.min(box1_x2, box2_x2) # ! (B, 7, 7, 1)
    y2 = torch.min(box1_y2, box2_y2) # ! (B, 7, 7, 1)

    # .clamp(0) is for the case when they do not intersect
    # ! 비정상적인 intersection 영역을 clamping 해줌
    # ! (x2 - x1).clamp(0) = 너비가 0보다 작은 경우 0으로 대체 
    # ! Intersection을 계산
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) # ! (B, 7, 7, 1)

    # ! Union 계산을 위해 예측된 bbox와 gt bbox의 크기 계산
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1)) # ! (B, 7, 7, 1)
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1)) # ! (B, 7, 7, 1)

    # ! IoU 계산 후 출력
    # ! intersection / (box1_area + box2_area - intersection + 1e-6) = (B, 7, 7, 1)
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    # ! NMS를 수행하는 code
    # ! bboxes        = (49, 6) -> 7x7 grid의 모든 grid cell 위치에서 예측된 class, confidence score, 최적 bbox 좌표 (x, y, w, h)
    # ! iou_threshold = 0.5
    # ! threshold     = 0.4
    # ! box_format    = "midpoint"

    assert type(bboxes) == list

    # ! Confidence score가 0.5 이상인 bboxex만 고려
    # ! n개라고 가정
    bboxes = [box for box in bboxes if box[1] > threshold] # ! (n, 6)

    # ! bboxes를 confidence score를 기준으로 내림차순 정렬
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # ! (n, 6)

    # ! 유지할 bboxes를 저장하기 위한 리스트 선언
    bboxes_after_nms = []

    while bboxes:

        # ! 현재 남아 있는 bboxes 중 가장 confidence score가 높은 bbox 추출
        chosen_box = bboxes.pop(0) # ! (6,)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]        # ! (i)  선택된 bbox와 같은 class이거나,
            or intersection_over_union(       # ! (ii) 선택된 bbox와의 IoU가 iou_threshold(0.5) 이상인 경우,
                torch.tensor(chosen_box[2:]), # ! 더 이상 고려하지 않음
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    # ! NMS 이후 남아 있는 bbox의 수가 p개라고 가정함
    # ! bboxes_after_nms = (p, 6)

    return bboxes_after_nms

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    # ! 주어진 IoU threshold 하에서 mAP를 계산
    # ! all_pred_boxes = (Qp, 7) -> 모든 Q개의 images에 대한 예측값이 (train_idx, s, c, x, y, w, h)의 형태로 저장되어 있음
    # ! all_true_boxes = (Qt, 7) -> 모든 Q개의 images에 대한 gt label이 (train_idx, s, c, x, y, w, h)의 형태로 저장되어 있음
    # ! iou_threshold  = 0.5
    # ! box_format     = "midpoint"

    # list storing all AP for respective classes
    # ! 모든 classes에 대한 AP를 저장하기 위한 리스트 선언 
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    # ! 모든 classes에 대해 각각 AP를 계산
    for c in range(num_classes): 
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        # ! 선택된 class에 속하는 예측값 식별
        # ! n개라고 가정
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection) # ! (n, 7)

        # ! 선택된 class에 속하는 gt label 식별
        # ! m개라고 가정
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box) # ! (m, 7)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        # ! 각 이미지마다 존재하는 gt label의 수 계산
        # ! 즉, 각 이미지에 주어진 class에 해당하는 gt label이 몇 개 있는지 계산
        # ! {0:3, 1:5, ...} -> 0번 이미지 3개, 1번 이미지 5개, ...
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        # ! Counter dictionary의 형식 변경
        # ! 만약 i번째 이미지에 j개의 gt label이 존재하는 경우,
        # ! dictionary의 i번째 key의 values를 j차원의 zero tensor로 변경
        # ! {0:3, 1:5, ...} -> {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0], ...}
        # ! 이러한 변경을 해주는 이유는 하나의 gt bbox에 대해 여러 개의 predictions가 존재할 수 있기 때문임
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        # ! n개의 예측값을 confidence score를 기준으로 내림차순 정렬
        detections.sort(key=lambda x: x[2], reverse=True) # ! (n, 7)

        # ! TP, FP 저장을 위한 tensor 정의
        TP = torch.zeros((len(detections))) # ! (n,)
        FP = torch.zeros((len(detections))) # ! (n,)
        total_true_bboxes = len(ground_truths) # ! gt label의 개수 = m
        
        # If none exists for this class then we can safely skip
        # ! 만약 해당 class의 gt label이 하나도 존재하지 않는 경우 계산을 skip
        if total_true_bboxes == 0:
            continue
        
        # ! 모든 정렬된 예측값에 대해 반복
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            # ! 선택된 예측 bbox와 같은 이미지에 속한 gt label만 선택
            # ! k개라고 가정
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ] # ! (k, 7)

            num_gts = len(ground_truth_img) # ! k
            best_iou = 0

            # ! 선택된 예측 bbox와 같은 이미지에 속한 gt label이 없는 경우 for문은 실행되지 않으며,
            # ! best_iou는 0이 되어 해당 예측은 FP로 분류됨
            for idx, gt in enumerate(ground_truth_img):
                # ! 모든 k개의 gt label과 IoU 계산
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                # ! 가장 IoU가 큰 gt label의 index 저장
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # ! IoU threshold를 만족하는 경우
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                # ! 만약 해당 gt label을 detect한 예측 bbox가 이전에 없었다면,
                # ! 현재 예측값은 TP로 분류됨
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1

                # ! 만약 해당 gt label을 detect한 예측 bbox가 이전에 있었다면,
                # ! 이는 현재 예측값보다 높은 confidence로 해당 gt label을 예측한 결과가 있었다는 뜻이므로,
                # ! 현재 예측값은 FP로 분류됨
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            # ! IoU threshold를 만족하지 못하는 경우 FP로 처리함
            else:
                FP[detection_idx] = 1

        # ! TP와 FP의 누적합(cumulative sum)을 계산함
        # ! 예측값은 confidence scores를 기준으로 내림차순 정렬되어 있으므로,
        # ! Confidence scores를 점차 낮춰갈 때 모든 confidence level에서의 TP와 FP를 한 번에 계산
        # ! ex. [1, 1, 0, 1, 0, ...] -> [1, 2, 2, 3, 3, ...]
        TP_cumsum = torch.cumsum(TP, dim=0) # ! (n,)
        FP_cumsum = torch.cumsum(FP, dim=0) # ! (n,)

        # ! recall 및 precision 계산
        recalls = TP_cumsum / (total_true_bboxes + epsilon) # ! (n,)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon)) # ! (n,)

        # ! PR curve 아래 영역 넓이(AP)를 계산(적분)하고 저장
        precisions = torch.cat((torch.tensor([1]), precisions)) # ! (n+1,)
        recalls = torch.cat((torch.tensor([0]), recalls)) # ! (n+1,)
        # torch.trapz for numerical integration
        # ! torch.trapz(precisions, recalls) -> scalar AP value
        average_precisions.append(torch.trapz(precisions, recalls))

    # ! mAP를 출력함
    # ! = 모든 classes의 AP 평균 = sum(average_precisions) / len(average_precisions)
    # ! 현재의 예시는 고정된 IoU threshold를 사용함 (i.e., 0.5)
    # ! 따라서 현재 계산된 mAP는 mAP@0.5임
    # ! 고정된 IoU threshold를 사용하는 대신 IoU threshold를 변경하면서 mAP를 구하고, 이를 평균한 mAP를 사용할 수도 있음
    # ! e.g., mAP@0.5:0.05:0.95 -> IoU threshold를 0.5에서 0.95까지 0.05씩 증가시키면서 각각 평가했을 때 평균적인 mAP 값

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):

        # ! x      = (B, 3, 448, 448) -> 입력 이미지
        # ! labels = (B, 7, 7, 30)    -> 입력 이미지에 대한 target matrix

        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            # ! Model output 계산
            # ! model.Yolov1 참조
            predictions = model(x)
            # ! predictions = (B, 1470) -> 모델 예측값

        batch_size = x.shape[0] # ! B

        # ! gt bboxes 변환
        # ! 예측/gt bboxes는 grid 혹은 grid cell에 대해 상대적으로 표현되어 있음
        # ! 이를 원본 이미지 차원의 값으로 변경해줌
        # ! 또한 각 grid cell마다 최적의 예측값을 선정하거나, 할당된 gt label을 식별하는 연산 수행
        # ! labels = (B, S, S, 30)
        true_bboxes = cellboxes_to_boxes(labels)
        # ! all_bboxes = (B, 49, 6) -> 모든 batch, 모든 grid cell 위치에서의 할당된 gt bbox 정보를 담은 3차원 리스트

        # ! 예측 bboxes 변환
        # ! predictions = (B, 1470) 
        bboxes = cellboxes_to_boxes(predictions)
        # ! all_bboxes = (B, 49, 6) -> 모든 batch, 모든 grid cell 위치에서의 최적 예측 bbox 정보를 담은 3차원 리스트
        
        for idx in range(batch_size):

            # ! 예측된 bboxes에 대해 NMS 적용
            # ! bboxes[idx]   = (49, 6)
            # ! iou_threshold = 0.5
            # ! threshold     = 0.4
            # ! box_format    = "midpoint"
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )
            # ! nms_boxes = (p, 6) -> NMS 적용 이후 남아 있는 p개의 bboxes


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            # ! m개의 예측 bboxes에 train_idx를 패딩하고 저장함
            # ! train_idx는 이미지 index를 의미
            for nms_box in nms_boxes:
                # ! [train_idx] + nms_box = [train_idx, s, c, x, y, w, h]
                all_pred_boxes.append([train_idx] + nms_box)

            # ! 49개의 gt bboxes에 train_idx를 패딩하고 저장함
            # ! train_idx는 이미지 index를 의미
            # ! 이때 gt confidence score가 0.4보다 큰 경우만 저장함
            # ! 즉, 할당된 object가 있는 grid cell의 gt label (s, c, x, y, w, h)만 저장함
            # ! t개라고 가정함
            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    # ! [train_idx] + box = [train_idx, s, c, x, y, w, h]
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    
    # ! 모든 Q개의 이미지 마다 p개의 예측값, t개의 gt label이 저장되었다고 가정함
    # ! 최종적으로 모든 Q개의 images에 대한 예측값, gt label이 (train_idx, s, c, x, y, w, h)의 형태로 출력됨
    # ! 이때 train_idx는 예측 혹은 gt label이 속하는 이미지의 index임
    # ! all_pred_boxes = (Qp, 7)
    # ! all_true_boxes = (Qt, 7)

    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    # ! 예측/gt bboxes는 grid 혹은 grid cell에 대해 상대적으로 표현되어 있음
    # ! 이를 원본 이미지 차원의 값으로 변경해줌

    # ! predictions = (B, 1470) or (B, 7, 7, 30) -> 모델 예측값 or gt label

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0] # ! B
    predictions = predictions.reshape(batch_size, 7, 7, 30) # ! (B, 7, 7, 30)
    bboxes1 = predictions[..., 21:25] # ! 첫 번째 bbox 좌표 (B, 7, 7, 4)
    bboxes2 = predictions[..., 26:30] # ! 두 번째 bbox 좌표 (B, 7, 7, 4)

    # ! 두 bbox의 confidence score를 concat함
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    ) # ! (2, B, 7, 7)

    # ! 각 grid cell에서 두 bbox 중 confidence score가 더 큰 bbox 식별
    # ! 만약 predictions가 모델 예측값인 경우 object가 있을 가능성이 더 큰 box를 식별하고,
    # ! predictions가 gt label인 경우 정답 bbox를 선택 (grid cell에 정답 bbox가 없는 경우도 존재 (= 할당된 object가 없는 경우))
    best_box = scores.argmax(0).unsqueeze(-1)
    # ! best_box = (B, 7, 7, 1)
    # ! index가 0이면 첫 번째 bbox가 선택되고, 1이면 두 번째 bbox가 선택됨을 의미

    # ! 각 grid cell에서 두 bbox 중 선택된 bbox의 좌표만 가져옴
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2 # ! (B, 7, 7, 4)

    # ! 좌표 변환을 위한 cell index 정의
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1) # ! (B, 7, 7, 1)

    # ! 현재 best_boxes를 구성하는 x, y는 해당 bbox가 속하는 grid cell 내에서의 상대적인 위치로 표현되어 있음 (0-1)
    # ! best_boxes[..., :1] = S * (x,y) - cell_indices
    # ! 이를 다시 원본 이미지에 대해 normalize된 좌표로 변경 (0-1)
    # ! (x,y) = (x + cell_indices) / S
    x = 1 / S * (best_boxes[..., :1] + cell_indices) # ! (B, 7, 7, 1)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3)) # ! (B, 7, 7, 1)

    # ! 마찬가지로 현재 best_boxes를 구성하는 w, h는 SxS(7x7) grid size를 기준으로 계산한 bbox의 상대적 width, height임
    # ! best_boxes[..., 2:4] = S * (w,h)
    # ! 이를 다시 원본 이미지에 대해 normalize된 w, h로 변경 (0-1)
    # ! (w,h) = (w,h) / S
    w_y = 1 / S * best_boxes[..., 2:4] # ! (B, 7, 7, 2)

    # ! 변환 결과를 저장
    converted_bboxes = torch.cat((x, y, w_y), dim=-1) # ! (B, 7, 7, 4)

    # ! 각 grid cell의 최적 class 계산
    # ! 만약 predictions가 모델 예측값인 경우 예측된 object class를 의미하고,
    # ! predictions가 gt label인 경우 정답 object class를 의미함 (grid cell에 정답 object class가 없는 경우도 존재 (= 할당된 object가 없는 경우))
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    # ! predicted_class = (B, 7, 7, 1) -> 최적 class의 index를 담고 있음

    # ! 두 bbox를 모두 고려했을 때 최대 confidence score를 저장
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    ) # ! (B, 7, 7, 1)

    # ! 변환된 예측값을 최종 저장
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    # ! (B, 7, 7, 6)

    # ! converted_preds = (B, 7, 7, 6) -> 모든 batch, 모든 grid cell 위치에서의 할당된/예측된 class label, 할당된/예측된 confidence score, 할당된/예측된 bbox 좌표가
    # ! (s, c, x, y, w, h)의 형태로 저장되어 있음

    return converted_preds


def cellboxes_to_boxes(out, S=7):

    # ! out = (B, 1470) or (B, 7, 7, 30) -> 모델 예측값 or gt label

    # ! 예측/gt bboxes는 grid 혹은 grid cell에 대해 상대적으로 표현되어 있음
    # ! 이를 원본 이미지 차원의 값으로 변경해줌
    # ! 또한 각 grid cell마다 최적의 예측값을 선정하거나, 할당된 gt label을 식별하는 연산 수행
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    # ! converted_pred = (B, 7, 7, 6) -> 모든 batch, 모든 grid cell 위치에서의 할당된/예측된 class label, 할당된/예측된 confidence score, 할당된/예측된 bbox 좌표가
    # ! (s, c, x, y, w, h)의 형태로 저장되어 있음
    # ! converted_pred.reshape(out.shape[0], S * S, -1) = (B, 49, 6)

    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]): # ! 모든 batch에 대해 반복
        bboxes = []

        for bbox_idx in range(S * S): # ! 모든 grid cell 위치에서

            # ! bbox 정보 추출 후 저장
            # ! [s, c, x, y, w, h]
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])

        # ! bboxes = (49, 6) -> 한 batch 내의 49개의 grid cell의 bbox 정보를 담은 2차원 리스트

        all_bboxes.append(bboxes)
        
    # ! all_bboxes = (B, 49, 6) -> 모든 batch, 모든 grid cell 위치에서의 최적 예측 or 할당된 gt bbox 정보를 담은 3차원 리스트

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
