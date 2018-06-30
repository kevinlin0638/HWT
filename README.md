# 機器學習作業 - Linear Classify


## 一、Prepare Before Excute Program
在開始執行前,請先import library.
  ```python
	import matplotlib.pyplot as plt
	from sklearn import datasets, svm, metrics
  ```

  
## 二、Load Picture into your program 

  - 將圖片由 datasets 讀入專案中
  ```python
        images_and_labels = list(zip(digits.images, digits.target))
	for index, (image, label) in enumerate(images_and_labels[:4]):
		plt.subplot(2, 4, index + 1)
		plt.axis('off')
		plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Training: %i' % label)
  ```

## 三、降低圖片維度 並 建立 Classifier
  
  ```python
        my_samples = len(digits.images)
		data = digits.images.reshape((my_samples, -1))
		
		# 建立一個 classifier: a support vector classifier
		classifier = svm.SVC(gamma=0.001)
  ```


## 四、學習與預測
[//]:U2FsdGVkX1+S0wU/4R6RatUoEm8KT+cRx05NtMHy2bq49ne9ep9nY985c6WcJAdo
先附上程式碼:

  ```python
	# 將前一半的資料拿來學習
	classifier.fit(data[:my_samples // 2], digits.target[:my_samples // 2])

	# 答案與預測答案:
	expected = digits.target[my_samples // 2:]
	predicted = classifier.predict(data[my_samples // 2:])
  ```

## 五、預測結果顯示
  
  ```
	  # 預測結果
	images_and_predictions = list(zip(digits.images[my_samples // 2:], predicted))
	for index, (image, prediction) in enumerate(images_and_predictions[:4]):
		plt.subplot(2, 4, index + 5)
		plt.axis('off')
		plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Prediction: %i' % prediction)

	plt.show()
  ```
  
[![N|Solid](https://i.imgur.com/fVvqxhA.jpg)](https://github.com/kevinlin0638)  
  執行結束示意圖  ↑

## 六、心得
	> 學習到了如何使用 python 來進行 手寫辨識的解題 的解題
	> 用sklearn真的很方便，使用精巧的程式碼即可完成

