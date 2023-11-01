# AIOT_Final

Step0：環境建置
  train與test：
  conda create --name tensorflow python=3.9
  conda activate tensorflow
  conda install jupyter notebook
  pip install tensorflow==2.5
  pip install opencv-python
  pip install opencv_contrib_python
  
  
  tello：
  使用pycharmr加入以下套件
  djitellopy
  opencv-python
  opencv-contrib-python


Step1：
使用train.ipynb來進行訓練
需要準備兩項 1. label.text 2.圖片檔(圖片檔解析度需一致)
![image](https://user-images.githubusercontent.com/74865648/210959000-5b89a993-bdec-4371-98e7-66ca7dba9ebb.png)
![image](https://user-images.githubusercontent.com/74865648/210959034-57e37640-1e87-4b05-8baf-cc9f324df90e.png)
接著會存在mnist_knn.xml

Step2：
接著可以用鏡頭做測試(test.ipynb)

Step3：
最後接上tello並使用tello.ipynb做測試

![圖片1](https://github.com/CiouQQ/AIOT_Final/assets/74865648/fca5fb21-cef1-4110-b7a4-48e27f9c1707)

