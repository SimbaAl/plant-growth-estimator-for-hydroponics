# plant-growth-estimator-for-hydroponics
Masters Project: Development of a machine learning plant growth estimator for hydroponics
Three Approaches of coming up with a plant growth estimator were developed and tested: (1) was the traditional machine learning approch where a feature extractor was designed using a combination of Gabor filtes, morphological operators, Sobel filter and A classification Algorithms (SVM and/ XGBOost), the second approach was using the transfer learning models, third approach was using a pretrained network as a feature extractor then use a machine learning model SVM of XGBoost for classification. 
Method 1 ( is the machine learning method with no morphological operators of opening and closing)
Method 2 (is the machine learning method with morphological operators)
Transfer learning models used are VGG16, VGG19 and RESNET-50
VGG16, VGG19 and RESNET-50 is combined with SVM and XGBoost in different models

Execution Time is measured in all the experiments
The conclusion from these experiments is that the third approach of using the convolutional layeys of a pretrained model and a classification algorithm produces the best resutls in terms of prediction time and acccuracy. 

