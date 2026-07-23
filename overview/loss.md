## object detection

- classification
    - binary cross entropy loss: given the sigmoid $p \in \mathbb{R^C}$, $BCE= -\sum\limits_{i=1}^{C} \left( y_i \log(p_i) + (1-y_i) \log(1-p_i) \right)$

- Bounding box
    - IoU loss, $IoU=\frac{B_{pred} \cap B_{true}}{B_{pred} \cup B_{true}}$, so that $L_{IoU} = 1 - IoU$
    - Distribution Focal Loss, with a distribution $p_1, \cdots, p_N$ with N=16, $y_1=1, \cdots, y_N=N$, $L_{DFL} = - \left( (y_{i+1} - y) \log(p_i) + (y - y_i) \log(p_{i+1}) \right)$



## semantic segmentation 
suppose we have pixels i=1, ..., N and classes c=1, ..., C, given ground truth label vector $y_i \in \mathbb{R}^C$ as a one-hot vector, and predicted probability vector $p_i \in \mathbb{R}^C$ after softmax transformation
- Pixel-wise Cross Entropy (CE) Loss, $L_{CE} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})$
- Dice Loss, $L_{Dice, c} = 1 - \frac{2 \sum_{i=1}^{N} (p_{i,c} \cdot y_{i,c}) + \epsilon}{\sum_{i=1}^{N} p_{i,c} + \sum_{i=1}^{N} y_{i,c} + \epsilon}$, and $L_{Dice} = \frac{1}{C} \sum_{c=1}^{C} L_{Dice, c}$


## depth estimation
Suppose we have ground truth depth $d_i$ and predicted depth $d_i^*$ for pixels i=1, ..., N where i is the pixel index, and $\Delta_i=\log(d_i^*)-\log(d_i)$, $\nabla_x d_i=d(x+1,y)-d(x,y)$ and $\nabla_y d_i=d(x,y+1)-d(x,y)$ 
- Scale-Invariant Log Loss, $L_{SILog} = \frac{1}{N} \sum\limits_{i=1}^{N} \Delta_i^2 - \frac{\lambda}{N^2} (\sum\limits_{i=1}^{N} \Delta_i)^2$
- Spatial Gradient Loss, $L_{Grad} = \frac{1}{N} \sum\limits_{i=1}^{N} \left( |\nabla_x d_i^* - \nabla_x d_i| + |\nabla_y d_i^* - \nabla_y d_i| \right)$