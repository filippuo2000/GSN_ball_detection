import streamlit as st
import cv2
import numpy as np
import torch
import os
from TrackNet import TrackNet
import tempfile
import torchvision.transforms as transforms
from torchvision.io import read_image
from TrackNetInference import video_writer, img_dir_writer

model = TrackNet()
checkpoint = torch.load(
    "/Users/Filip/PycharmProjects/GSN_track/GSN_ball_detection/checkpoints/model-epoch=20-val_loss=0.016e36f8152809114ccc9d56ab71dc0d19408159ee09b166b724337e9abfc13c07.ckpt",
    map_location=torch.device('cpu'))
state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Streamlit app
st.title('Tennis Ball Detection')
st.write('Upload a tennis video to detect the ball.')

# Upload video
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Create a temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    video_writer(model, tfile.name, 'output_video.mp4', True)
    st.video('output_video.mp4')
