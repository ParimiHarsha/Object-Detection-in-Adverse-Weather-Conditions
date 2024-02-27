# Object Detection Using YOLOv8

The goal of this project is to create an object detection model that is robust to adverse weather conditions (e.g. rain, fog, night). 

## Choice of Data

Initially, I started by searching if there is any work that has already been done on this. And I came across a couple of research papers related to this:

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10611033/#B67-sensors-23-08471
- https://openaccess.thecvf.com/content/ICCV2021/papers/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.pdf

After going through the above papers and researching further, I found the following datasets suitable for our analysis:

1. ACDC (The Adverse Conditions Dataset with Correspondences)
    1. This dataset has 4006 camera images from Zurich (Switzerland) recorded in four weather conditions: rain, fog, snow, and night. The ACDC has all photos with one of any of the weather features and 4006 images that are evenly distributed for each weather characteristic
2. DAWN (Vehicle Detection in Adverse Weather Nature)
    1. This dataset contains 1027 photos gathered from web searches on Google and Bing, was another highly relevant dataset. However, it has extremely harsh weather qualities, which can serve as a real-world example for training and testing under adverse conditions. It also includes several sand storm images that offer distinctive aspects compared to the other datasets.

After going through the datasets, I decided to work with the ACDC dataset as it had more images. But the DAWN dataset can later be used to test the performance of the trained model.

ACDC dataset has train, validation and test sets predefined. Here are a few images 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/94e34583-542e-4863-9bdb-d02b6266d819/d6f86f30-9e87-4655-9711-8b5e305f62ce/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/94e34583-542e-4863-9bdb-d02b6266d819/06862cf5-4003-4bb0-9ae8-738a23e5302d/Untitled.png)

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

![Class distribution in the train data](https://prod-files-secure.s3.us-west-2.amazonaws.com/94e34583-542e-4863-9bdb-d02b6266d819/d20a61ce-28c9-44a4-8767-598654318002/Untitled.png)

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

While creating the labels for the images we need to change the label ids, e.g cars are labelled as ‘26’ in the raw data but we relabelled it as ‘0’ for our model. We can keep adding new label ids as we require for detection.

## Model Training

For this exercise I have decided to use the small size model `yolov8n.yaml` as we are dealing with a fairly small dataset. 

Initially tried training the above model from scratch, but it was taking too long to train. The results could have been better if I trained it for longer time.

Then, started to work with pre-trained weights use transfer learning to make it dataset specific. This model gave good results while training for a reasonable amount of time.

Here are the validation predictions

![Actual validation labels](https://prod-files-secure.s3.us-west-2.amazonaws.com/94e34583-542e-4863-9bdb-d02b6266d819/2d07e531-7f96-4fe1-b91d-3f64c05ee19f/Untitled.jpeg)

Actual validation labels

![Validation predictions](https://prod-files-secure.s3.us-west-2.amazonaws.com/94e34583-542e-4863-9bdb-d02b6266d819/d2de6a9d-cfee-402b-b8e3-ef84503b0a26/Untitled.jpeg)

Validation predictions

After training the model on test data I have tried to test the model’s perfomance in real time by using a video of a car driving in snow.

Talk about the evaluation metrics map50

references
https://kikaben.com/yolov5-transfer-learning-dogs-cats/

https://acdc.vision.ee.ethz.ch/download

https://openaccess.thecvf.com/content/ICCV2021/papers/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.pdf

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10611033/#B67-sensors-23-08471

https://docs.ultralytics.com/modes/train/
