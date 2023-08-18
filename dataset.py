"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S # ! 7
        self.B = B # ! 2
        self.C = C # ! 20

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        # ! 이미지에서 bbox, class 정보 추출
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            # ! image에 대해서만 transform 적용됨 (train.Compose 참고)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        # ! 주어진 이미지의 예측 target 생성
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            # ! x, y는 0-1로 normalize되어 있음
            # ! i, j는 7x7 grid cell 상의 정수 좌표임
            # ! 즉, i, j는 7x7의 grid 중 어떤 grid cell에 해당 bbox가 할당되는지를 의미함
            i, j = int(self.S * y), int(self.S * x)
            # ! x_cell, y_cell은 bbox의 중심좌표를 해당 bbox가 속하는 grid_cell 내에서의 상대적인 위치로 나타냄 (0-1)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            # ! width_cell, height_cell은 7x7 grid cell을 기준으로 계산한 bbox의 상대적 width, height임
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            # ! 한 grid cell에 하나의 class만 할당되므로 선택된 (i,j) cell에 할당된 class가 없는 경우만 처리
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                # ! 정답 bbox인 첫 번째 bbox에 object가 있다는 것을 저장
                label_matrix[i, j, 20] = 1

                # Box coordinates
                # ! 정답 bbox의 좌표 저장
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                # ! 20개의 class 중 해당하는 gt class label 저장
                # ! one-hot form으로 구성되어 있음
                label_matrix[i, j, class_label] = 1

        # ! image        = (3, 448, 448) -> 입력 이미지
        # ! label_matrix = (S, S, 30)    -> 입력 이미지에 대한 target matrix

        return image, label_matrix
