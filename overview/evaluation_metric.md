# depth estimation
suppose ground truth depth is marked as (d_1, d_2, ..., d_n) and predicted depth is marked as (d_1^*, d_2^*, ..., d_n^*) among all images 
## absolute relative error(Abs Rel)

$Abs Rel = \frac{1}{n} \sum_{i=1}^{n} \frac{|d_i - d_i^*|}{d_i}$

## root mean squared error(RMSE)

$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (d_i - d_i^*)^2}$


## Threshold Accuracy ($\delta < 1.25$)
$ Thres\_Acc = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}_{\max\left(\frac{d_i}{d_i^*}, \frac{d_i^*}{d_i}\right) < \delta}$

# semantic segmentation
mean intersection over union (mIoU)

for each class, compute the intersection over union (IoU) as follows:
$$
IoU = \frac{TP}{TP + FP + FN}
$$

then compute the mean IoU across all classes:
$$
mIoU = \frac{1}{C} \sum_{i=1}^{C} IoU_i
$$


# object detection
- mean average precision (mAP) @0.5
- mean average precision (mAP) @0.75
- mean average precision (mAP) @0.5:0.95

## Average Precision(AP) @k computation
For a specific class(e.g., car), given all generated bounding boxes during the inference stage, define the positive detection if 
- IoU is greater than a certain threshold $k$  
- confidence score is greater than a certain threshold $\epsilon$


Gradually decrease the confidence score threshold $\epsilon$ from 1 to 0 to increase the recall and record the corresponding precision($Precision=\frac{TP}{P}=\frac{TP}{TP + FP}$), and finally derive the Precision-Recall (PR) curve, and then compute the integral of the PR curve to get the average precision (AP) for each class. 

## mean Average Precision(mAP) @k computation
mAP@k computation procedure, with $C$ classes and the average precision for each class $AP_i@k$, is computed as follows:
$$
mAP@k = \frac{1}{C} \sum_{i=1}^{C} AP_i@k
$$


## overall mAP@0.5:0.95
Finally, to evaluate the general model performance under different IoU thresholds, we can compute the overall mAP@0.5:0.95 by averaging the mAP@k values for $k$ ranging from 0.5 to 0.95 with a step size of 0.05:
$$
mAP@0.5:0.95 = \frac{1}{10} \sum_{k=0.5}^{0.95} mAP@k
$$  
