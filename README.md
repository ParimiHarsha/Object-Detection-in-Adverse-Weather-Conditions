# Object Detection Using YOLOv8

The goal of this project is to create an object detection model that is robust to adverse weather conditions (e.g. rain, fog, night). 

## Literature Review

Initially, I started by searching if there is any work that has already been done on this. And I came across a couple of research papers related to this:

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10611033/#B67-sensors-23-08471
- https://openaccess.thecvf.com/content/ICCV2021/papers/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.pdf

Mitigating domain discrepancies from normal weather to adverse climate in vision neural networks involves several strategies aimed at improving the model's robustness and generalization across different weather conditions. 

- Here are some approaches to mitigate domain discrepancies:
    1. **Data Augmentation**:
        - Augmenting the dataset with diverse samples from both normal and adverse weather conditions helps expose the model to a wider range of scenarios.
        - Techniques such as adding simulated fog, rain, snow, and haze to images can help the model learn to recognize objects under adverse weather conditions.
    2. **Transfer Learning**:
        - Transfer learning involves using a pre-trained model trained on a source domain (e.g., normal weather conditions) and fine-tuning it on the target domain (e.g., adverse weather conditions).
        - By leveraging features learned from the source domain, the model can adapt more quickly to the target domain while mitigating domain discrepancies.
    3. **Domain Adaptation**:
        - Domain adaptation techniques aim to align feature distributions between different domains (e.g., normal weather and adverse climate) to reduce domain discrepancies.
        - Adversarial training, where a domain discriminator is trained to distinguish between source and target domain features, can help align the feature distributions.
        - Domain-specific regularization techniques can also be employed to encourage the model to learn domain-invariant features.
    4. **Data Preprocessing**:
        - Preprocessing techniques such as contrast enhancement, de-noising, and haze removal can improve the quality of input images and reduce the impact of adverse weather conditions on the model's performance.
        - Data preprocessing methods specifically tailored to handle adverse weather conditions can help mitigate domain discrepancies and improve the model's robustness.
    5. **Domain Randomization**:
        - During training, expose the model to a wide variety of weather conditions, including both normal and adverse climates.
        - Randomizing weather conditions during training helps the model learn to generalize across different domains and reduces overfitting to specific conditions.
    6. **Robust Model Architecture**:
        - Utilize robust neural network architectures that are capable of capturing and representing features across different weather conditions.
        - Architectures such as convolutional neural networks (CNNs) with skip connections, feature pyramids, and attention mechanisms can help improve the model's ability to handle variations in weather conditions.

I have chosen to implement the transfer learning approach which was the most viable because:

1. **Utilization of Pre-trained Models:** It utilizes pre-trained models that hvae been trained on large-scale datasets in normal weather conditions. These pre-trained models have already learned generic features and patterns that are transferable across different domains.
2. **Faster Training and Convergence:** By starting with a pre-trained model, transfer learning reduces the training time and computational resources required to adapt the model to the target domain. The model initializes with weights that are already optimized for extracting relevant features, which can accelerate the convergence of training on the new dataset.

It also has other advantages like Reduced Risk of Overfitting and Effective Feature Extraction.

## Choice of Data

After going through the above papers and researching further, I found the following datasets suitable for our analysis:

1. ACDC (The Adverse Conditions Dataset with Correspondences)
    1. This dataset has 4006 camera images from Zurich (Switzerland) recorded in four weather conditions: rain, fog, snow, and night. The ACDC has all photos with one of any of the weather features and 4006 images that are evenly distributed for each weather characteristic
2. DAWN (Vehicle Detection in Adverse Weather Nature)
    1. This dataset contains 1027 photos gathered from web searches on Google and Bing, was another highly relevant dataset. However, it has extremely harsh weather qualities, which can serve as a real-world example for training and testing under adverse conditions. It also includes several sand storm images that offer distinctive aspects compared to the other datasets.

After going through the datasets, I decided to work with the ACDC dataset as it had more images. But the DAWN dataset can later be used to test the performance of the trained model.

ACDC dataset has train, validation and test sets predefined.

It has the following labels and ids

```
class_names = {
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle',
}
```

Class distribution in the train data

As the data is mostly contained of car labels, I have decided to only detect cars for this assignment, this also helps in training the model faster and is more accurate. 

## Model Selection

Decided to use YOLOv8 for this project. As it is faster than other models. It processes images in a single pass, making it well-suited for real-time applications. It is also fairly simple to implement. Here is the model architecture:

```python
# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  - [-1, 1, Conv, [64, 6, 2, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C3, [128]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C3, [256]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 9, C3, [512]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C3, [1024]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv5 v6.0 head
head:
  - [-1, 1, Conv, [512, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C3, [512, False]]  # 13

  - [-1, 1, Conv, [256, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C3, [256, False]]  # 17 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C3, [512, False]]  # 20 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C3, [1024, False]]  # 23 (P5/32-large)

  - [[17, 20, 23], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

## Data Preprocessing

The data preprocessing involves 2 main steps:

1. Populating the training images in `data/images` directory from all the weather condition files.
2. Creating the labels for the images in the YOLOv8 format and populating the `data/labels` directory.

While creating the labels for the images we need to change the label ids, e.g cars are labelled as ‘26’ in the raw data but we relabelled it as ‘0’ for our model. We can keep adding new label ids as we require. 

## Model Training

For this exercise I have decided to use the small size model `yolov8n.yaml` (3.2M params) as we are dealing with a fairly small dataset. 

Initially tried training the above model from scratch, but it was taking too long to train. The results could have been better if I trained it for longer time.

Then, started to work with pre-trained weights use transfer learning to make it dataset specific. This model gave good results while training for a reasonable amount of time.

## Testing the Model

I have then used the trained model to predict on the test data. Here are a few predicted images

After getting satisfactory results on the test data, I have tried to test the model’s performance in real time by using a video of a car driving in snow. (DEMO).

## Challenges

- Understanding the dataset: It took me time to go through the dataset structure(images and labels) and convert it into the way yolo format.
- Model Training Times: I initially spent a lot of time trying to train the model from scratch, thinking that it would be better. But then I realized that we need to utilize the feature extractors of the pre-trained model and fine tune it on our dataset

## References
https://openaccess.thecvf.com/content/ICCV2021/papers/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.pdf

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10611033/#B67-sensors-23-08471

https://docs.ultralytics.com/modes/train/
