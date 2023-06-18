import streamlit as st
import os
from fastai.vision.all import *

path = os.path.dirname(os.path.abspath(__file__))
model1_path = os.path.join(path, 'resnet18.pkl')
model2_path = os.path.join(path, 'vgg.pkl')

learn1 = load_learner(model1_path)
learn2 = load_learner(model2_path)

uploaded_file = st.file_uploader("上传一张四大名楼的图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = PILImage.create(uploaded_file)
    except:
        st.error("Invalid Image")
    else:
        st.image(img.to_thumb(500, 500), caption='Your Image')
        pred1, pred1_idx, probs1 = learn1.predict(img)
        pred2, pred2_idx, probs2 = learn2.predict(img)
        st.write(f"resent18 识别结果: {pred1}; 准确率"f": {probs1[pred1_idx]:.04f}")
        st.write(f"vgg 识别结果: {pred2}; 准确率: {probs2[pred2_idx]:.04f}")
        # 如果两个模型的预测概率都低于0.8提示上传的图片有问题
        if probs1[pred1_idx] < 0.9 and probs2[pred2_idx] < 0.7:
            st.warning("你上传的图片识别准确率较低，有可能不是四大名楼的照片")

    # 关闭文件上传对象，释放内存空间
    uploaded_file.close()