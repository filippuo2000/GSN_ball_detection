import cv2
import numpy as np
import torch
import os
from TrackNet import TrackNet
import torchvision.transforms as transforms
from torchvision.io import read_image

def postprocess_output(feature_map: torch.Tensor, scale=1):
    _, feature_map = torch.max(feature_map, dim=1)
    feature_map = feature_map.float()
    feature_map = feature_map.cpu()
    locations = torch.zeros((2, feature_map.shape[0])) - 1
    feature_map = torch.transpose(feature_map, 1, 2)
    feature_map = feature_map.detach().numpy()
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map[:], 127, 255, cv2.THRESH_BINARY)

    for i in range(feature_map.shape[0]):
        heatmap_i = cv2.GaussianBlur(heatmap[i], (5, 5), 0, 0)
        circles = cv2.HoughCircles(heatmap_i, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=100, param2=0.9,
                                   minRadius=1,
                                   maxRadius=15)
        x, y = -10, -10
        if circles is not None:
            if len(circles) == 1:
                x = circles[0][0][1] * scale
                y = circles[0][0][0] * scale
        locations[0][i], locations[1][i] = x, y

    return locations

def preprocessing(img: torch.Tensor):
    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(360, 640), mode='bilinear', align_corners=False).squeeze(0).float()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],  # RGB mean
                                     std=[0.5, 0.5, 0.5])
    img=normalize(img)
    img = torch.unsqueeze(img, dim=0)
    return img

# FOR Video Input
def video_writer(model, video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30, (640, 360))

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, ball_img = cap.read()
        cv2.imwrite("temp/img.jpg", ball_img)
        if ret == True:
            frame = read_image("temp/img.jpg")
            ball_img = cv2.resize(ball_img, (640, 360))
            frame = preprocessing(frame)
            with torch.no_grad():
                output = model.forward(frame)
                pos = postprocess_output(output, scale=1)
            ball_img = cv2.circle(ball_img, (int(pos[0].item()), int(pos[1].item())), 3, (0, 255, 0), 2)
            out.write(ball_img)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# For input consisting of images in a directory
def img_dir_writer(model, imgs_dir_path, out_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30, (640, 360))
    # video_writer(cap, out, model)
    imgs_dir = imgs_dir_path
    imgs = os.listdir(imgs_dir_path)

    for img_path in sorted(imgs):
        print(img_path)
        ball_img = cv2.imread(os.path.join(imgs_dir, img_path))
        img = read_image(os.path.join(imgs_dir, img_path))
        ball_img = cv2.resize(ball_img, (640, 360))
        img = preprocessing(img)

        with torch.no_grad():
            output = model.forward(img)
            pos = postprocess_output(output, scale=1)
        ball_img = cv2.circle(ball_img, (int(pos[0].item()), int(pos[1].item())), 3, (0, 255, 0), 2)
        out.write(ball_img)

def main():
    model = TrackNet()
    checkpoint = torch.load(
        "/Users/Filip/PycharmProjects/GSN_track/GSN_ball_detection/checkpoints/model-epoch=20-val_loss=0.016e36f8152809114ccc9d56ab71dc0d19408159ee09b166b724337e9abfc13c07.ckpt",
        map_location=torch.device('cpu'))
    state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Ex. use
    img_dir_writer(model, "/Users/Filip/Downloads/Dataset/game9/Clip6", "videos/final_2.mp4")


if __name__ == '__main__':
    main()