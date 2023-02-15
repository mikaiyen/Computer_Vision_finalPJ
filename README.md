# Computer_Vision_finalPJ
## 1.	MOTIVATION  
CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) is used by websites to differentiate between real users and bots.
These challenges are omnipresent on the dark web to protect platforms from the constant DDoS attacks that competing platforms launch against each other.

## 2.	GOAL
Recognize input captcha image and output its’ correct code
Can further use on dark web crawler

## 3. DATASET
Dataset from Kaggle
 ![image](https://user-images.githubusercontent.com/119439749/218959859-c6a85b16-fb95-4894-a826-9fb2786777d7.png)

## 4. DATA PREPROCESSING 
![image](https://user-images.githubusercontent.com/119439749/218959938-4aeecf71-d220-4242-95d5-5b4d4a09d6c0.png)

## 5.	MODEL
### (1) Use ReLU as activation function:
![image](https://user-images.githubusercontent.com/119439749/218960152-aad8b504-53a0-424f-a859-598d4de66699.png)
  1.1	calculation fast  
  1.2	accelerate the convergence speed  
  1.3	the problem of gradient disappearance is 	solved in the X>0 interval
 
![image](https://user-images.githubusercontent.com/119439749/218960346-fb27fa22-c3a9-4a03-a5fa-36e01a774eb6.png)

### (2)	Use  tf.keras.layers.Dropout, which helps prevent overfitting
### (3)	Use softmax as output layer activation function for multi-output classification

## 6.	MAIN PROGRAM
### 6.1 Backend: cvcaptcha.py
提供許多function供mainui呼叫，達成前後端分離
1.	create_model():	建造一個CNN model
2.	modelexist():	檢查是否有已經保存的model	
3.	preprocess_data():	資料(圖片)前處理
4.	show_train_history():	以圖像顯示訓練中train 和validation的accuracy和loss
5.	build_model(bsize,eps):	讓使用者輸入bsize(batch size)和eps(epoch)來訓練模型
6.	savemodel(newmodel):	將newmodel儲存為captcha.h5以供使用
7.	predict(filepath,usenewmodel):	使用者選擇圖片(filepath)，選擇要用哪個模型(usenewmodel)來預測，最後return capt 就是預測的captcha符號
### 6.2	Frontend: mainui.py(請從此程式執行!)
使用tkinter做GUI
![image](https://user-images.githubusercontent.com/119439749/218960747-8cff08a3-c530-498c-829d-bc6a2c901cc0.png)
1.	Choose data 按鈕:	供使用者選擇captcha的圖檔(在captcha-version-2-images資料夾內)
2.	Training 按鈕:	使用者在輸入batch size和epoch數字後(如不輸入則預設batch size=32; epoch=30)，按下此按鈕即可開始訓練模型，最後會呼叫show_train_history()顯示訓練的accuracy和loss，訓練結束會詢問使用者是否要保存模型，若為真，則執行savemodel(newmodel)
3.	Predict 按鈕:	檢查檔案路徑是否存在後，若為真，則在詢問使用者要使用的模型後(預設為batch size=32; epoch=30，檔名為defaultcaptcha.h5)，執行predict(filepath,usenewmodel)，在GUI上方顯示圖片，右方顯示預測結果Prediction
FILE PATH 不可為中文!會出錯!如果要測試請把路徑中文改成英文!

## 7.	RESULT
### 7.1 Result when use sigmoid as output layer:
![image](https://user-images.githubusercontent.com/119439749/218961016-0e53685a-7afb-4bc2-943f-f373f57ee65a.png)![image](https://user-images.githubusercontent.com/119439749/218961027-b9617411-3525-4d81-9336-a6c809b44457.png)
### 7.2 Result when use softmax as output layer:
![image](https://user-images.githubusercontent.com/119439749/218961050-7989313a-49c0-449b-a233-1bb4940ea7e2.png)![image](https://user-images.githubusercontent.com/119439749/218961063-37f328c2-84ac-410d-a536-8fe32f3c97c6.png)

