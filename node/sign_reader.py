#!/usr/bin/env python3

import cv2
from os.path import join
import os
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import time

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


bridge = CvBridge()
target_dim = (300, 200)
lastRead = time.time()


#==================================================

# Isolate the blue
def isolate_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    TargetHue = 240/2

    # Define the Hue range for blue (240 degrees is pure blue)
    # In OpenCV, Hue is scaled from 0-360 degrees to 0-180.
    # So, 240 degrees corresponds to 240/2 = 120.
    lower_blue = np.array([TargetHue-1, 50, 50])
    # Upper bound: Hue=130, Sat=255, Val=255
    upper_blue = np.array([TargetHue+1, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    return mask

def order_points(pts):
    """
    Orders coordinates in the order:
    top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# Find the contours
def isolate_countours(image):
    # 2. First Pass: Find everything
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 3. Find the absolute largest shape (The Outer Box)
        # This is your "Main Rectangle"
        outer_contour = max(contours, key=cv2.contourArea)

        # 4. Create a mask to "get rid of everything not in the contour"
        mask = np.zeros_like(image)

        # Draw the outer contour filled with white (255) to create a solid block
        cv2.drawContours(mask, [outer_contour], -1, 255, thickness=cv2.FILLED)

        # 5. Apply the mask to the original image
        # Now 'isolated_image' contains ONLY the main frame. All noise is black.
        isolated_image = cv2.bitwise_and(image, image, mask=mask)

        # 6. Second Pass: Find contours again on this clean image
        clean_contours, hierarchy = cv2.findContours(isolated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort these new contours by area (Largest to Smallest)
        # sorted_clean[0] = The Outer Edge (The limit of the shape)
        # sorted_clean[1] = The Inner Edge (The screen hole)
        sorted_clean = sorted(clean_contours, key=cv2.contourArea, reverse=True)

        return isolated_image

# Check conditions, do transform
def persp_transform(thresh):

    AREA_THRESH = 15000

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(sorted_contours) < 2:
        return -1 # failed
    second_largest_contour = sorted_contours[1]

    # debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    # print(f"Sign area: {cv2.contourArea(second_largest_contour)}")
    # cv2.drawContours(debug, [second_largest_contour], -1, (0, 255, 0), 2)
    # cv2.imshow(cv2.resize(debug, (300, 300)))

    if(cv2.contourArea(second_largest_contour) < AREA_THRESH):
      #print(f"Area: {cv2.contourArea(second_largest_contour)}")
      return -1 # failed

    # debug = thresh
    # cv2.drawContours(debug, [second_largest_contour], -1, 120, 2)
    # cv2_imshow(debug)

    # At the moment I'm just assiming that we get 4 corners
    approx = cv2.approxPolyDP(second_largest_contour, 0.02*cv2.arcLength(second_largest_contour, True), True)
    if(len(approx)!=4):
        return -1 # failed
    approx_ordered = order_points(approx.reshape(4, 2))

    # find the destination of the persepctive shift
    (tl, tr, br, bl) = approx_ordered
    # Calculate width of new image
    maxWidth = 300

    # Calculate height of new image
    maxHeight = 200
    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

    # Do the persepctive shift
    M = cv2.getPerspectiveTransform(approx_ordered, dst)
    shifted = cv2.warpPerspective(thresh, M, (maxWidth, maxHeight))
    return shifted

# Resize to consistend dimensions - may be irrelevant now
def resize_with_padding(image, target_size, pad_color=(0, 0, 0)):
    """
    Resizes an image to a target size while maintaining aspect ratio
    by adding padding (letterboxing).

    Args:
        image (numpy.ndarray): The input image (from your perspective transform).
        target_size (tuple): The desired output size (width, height).
        pad_color (tuple): RGB value for the padding. Default is black.

    Returns:
        numpy.ndarray: The resized and padded image ready for the CNN.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # 1. Calculate the scaling factor
    # We want the image to fit inside the target box, so we take the minimum scale
    scale_w = target_w / w
    scale_h = target_h / h
    scale = min(scale_w, scale_h)

    # 2. Compute new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 3. Resize the image
    # INTER_AREA is generally best for shrinking, INTER_LINEAR for enlarging
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # 4. Calculate padding amounts
    delta_w = target_w - new_w
    delta_h = target_h - new_h

    # Split the padding evenly (center the image)
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # 5. Apply the padding using copyMakeBorder
    # This is much faster than creating a blank canvas and pasting the image
    padded_image = cv2.copyMakeBorder(
        resized_image,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )

    return padded_image

# Crops the rectangularized sign to isolate the text on the bottom
# This is to be fed into the CRNN
def crop_text(img):
  '''
  Returns:
    - Cropped image
    - Text label
  '''

  edge_buffer = 7
  width = 300-2*edge_buffer
  height = 55

  # Area
  top_left_corner = (edge_buffer, 120)  # (x, y) coordinates of the top-left point
  bottom_right_corner = (edge_buffer + width, 120+height) # (x, y) coordinates of the bottom-right point

  # create a cropped image
  cropped_img = img[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]
  #cv2_imshow(cropped_img)

  return cropped_img

#==================================================

# CRNN class
class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256):
        super(CRNN, self).__init__()

        # We assume img_height is 64

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # H: 64 -> 32, W: 256 -> 128

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # H: 32 -> 16, W: 128 -> 64

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Block 4 (Vertical pooling only to keep sequence length long)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)), # H: 16 -> 8, W: 64 -> 64

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        # --- MATH UPDATE HERE ---
        # With input height 64, the final height after pooling is 8.
        self.cnn_output_height = 8
        self.cnn_output_channels = 512

        # Calculate input size for the Linear layer
        self.map_to_seq = nn.Linear(self.cnn_output_channels * self.cnn_output_height, map_to_seq_hidden)

        self.rnn = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(rnn_hidden * 2, num_class)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()

        # Standard reshape logic
        conv = conv.permute(0, 3, 1, 2)
        conv = conv.reshape(b, w, -1)

        conv = self.map_to_seq(conv)
        rnn_out, _ = self.rnn(conv)
        output = self.dense(rnn_out)

        # Prepare for CTC Loss
        output = output.permute(1, 0, 2)
        output = F.log_softmax(output, dim=2)
        return output

# Decoder (reads CRNN output and converts to a string)
def greedy_decoder(preds, alphabet):
    """
    preds: Tensor of shape (Time, Batch, Class) -> The output of the model
    alphabet: String of characters '012...Z'
    """
    # Take max probability at each time step
    # Shape: (Time, Batch)
    _, max_index = torch.max(preds, dim=2)

    # Transpose to (Batch, Time) for easier iteration
    max_index = max_index.transpose(1, 0)

    decoded_strings = []

    for i in range(max_index.size(0)): # Iterate over batch
        sequence = max_index[i]
        decoded_text = []
        prev_char_index = -1

        for index in sequence:
            index = index.item()
            # 0 is the blank token (assuming you set blank=0 in CTCLoss)
            if index != 0 and index != prev_char_index:
                decoded_text.append(alphabet[index - 1]) # -1 because 0 is blank
            prev_char_index = index

        decoded_strings.append(''.join(decoded_text))

    return decoded_strings

def predict_single_image(model, img, device, alphabet):

    # Check if image loaded correctly
    if img is None:
        print("Error: Image not found.")
        return

    # Resize to standard size (256x64)
    img = cv2.resize(img, (256, 64))

    # Normalize
    img = img / 255.0

    # Add dimensions to match model input: (Batch, Channel, Height, Width)
    # (1, 1, 64, 256)
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)

    # 2. Run Inference
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)

    # 3. Decode
    pred_text = greedy_decoder(pred, alphabet)[0] # Take first item of batch

    print(f"Prediction: {pred_text}")
    return pred_text

#==================================================

# Reads the image
# Returns if a sign can be read or not
def imageProcess(msg):

    global lastRead
    currRead = time.time()

    if(currRead - lastRead > 5):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        thresh = isolate_hue(img)
        contours = isolate_countours(thresh)
        transformed = persp_transform(contours)

        # cv2.imshow("contours", contours)
        # cv2.waitKey(1)

        # If sign found, read it and publish
        if not isinstance(transformed, int):
            lastRead = currRead
            text = crop_text(transformed)
            sign_text = predict_single_image(loaded_model, text, device, ALPHABET)
            text_pub.publish(sign_text)
            # cv2.imshow("the image", text)
            # cv2.waitKey(1)

            return True
        else:
            return False
    


if __name__ == '__main__':
    lastRead = time.time()
    rospy.init_node("sign_reader", anonymous=True)
    text_pub = rospy.Publisher("/text_read", String, queue_size=10)

    #* Load the model
    PATH = '/home/fizzer/ros_ws/src/robot_ctrl_pkg/node/crnn_ocr_model.pth'
    ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ" # Added space, removed numbers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Re-initialize the model structure
    # Make sure these params match what you trained with!
    loaded_model = CRNN(img_channel=1, img_height=64, img_width=256, num_class=len(ALPHABET)+1)

    # 2. Load the weights
    loaded_model.load_state_dict(torch.load(PATH, map_location=device))

    # 3. Set to Eval mode (Crucial! Disables dropout)
    loaded_model.to(device)
    loaded_model.eval()

    print("Model loaded and ready for testing!")

    rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, imageProcess)
    rospy.spin()

