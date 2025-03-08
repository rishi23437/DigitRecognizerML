# DigitRecognizerML
This model, trained on the MNIST database of handwrittend digits, recognizes and classifies the digits 0, 1 and 2. It involves:
* Maximum Likelihood Estimation (MLE) for parameter estimation.
* Principal Component Analysis (PCA) for dimensionality reduction.
* Fisher's Discriminant Analysis (FDA) for optimal projection.
* Linear and Quadratic Discriminant Analysis (LDA & QDA) for classification.


## Installation and Usage
* Clone the repository
``` bash
git clone https://github.com/rishi23437/DigitRecognizerML.git
cd DigitRecognizer
```
* Download the [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) from Kaggle and place it in the main directory.
* Run the main file
``` bash
python main.py
```


## Description
I compare the performance of this model in classifying the PCA and FDA-transformed datasets:
- PCA retains 95% variance - about 82 principle components.
- FDA maximizes the trace ratio of the between class and within class scatter matrices.
- I classify the resultant PCA and FDA data using LDA. Additionally, I also apply QDA on the FDA dataset.
Additionally, the project also displays 2D plots of the PCA and FDA-transformed datasets.


## Results
- LDA accuracy on PCA-transformed dataset: 95.67%.
- LDA accuracy on FDA-transformed dataset: 94.67%.
- QDA accuracy on PCA-transformed dataset: 97.00%.


## References
- [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
- The Lecture Slides of my Statistical machine Learning course.
