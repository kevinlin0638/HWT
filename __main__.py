# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# 讀取所需圖片
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# 降低維度
my_samples = len(digits.images)
data = digits.images.reshape((my_samples, -1))

# 建立一個 classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# 將前一半的資料拿來學習
classifier.fit(data[:my_samples // 2], digits.target[:my_samples // 2])

# 答案與預測答案:
expected = digits.target[my_samples // 2:]
predicted = classifier.predict(data[my_samples // 2:])

# 預測結果
images_and_predictions = list(zip(digits.images[my_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
