import cv2
import numpy as np

import time
import random

def random_digit_color(img ,bg_images):

    bg_image = random.choice(bg_images)
    bg_h, bg_w, _ = bg_image.shape
    img_h, img_w, _ = img.shape

    if bg_h < img_h or bg_w < img_w:
        raise ValueError("Background image is smaller than the input image")

    x = random.randint(0, bg_w - img_w)
    y = random.randint(0, bg_h - img_h)
    crop = bg_image[y:y + img_h, x:x + img_w]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(crop, crop, mask=mask_inv)
    digit_part = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.add(bg_part, digit_part)
    return img

def random_background(img ,bg_images):

    bg_image = random.choice(bg_images)
    bg_h, bg_w, _ = bg_image.shape
    img_h, img_w, _ = img.shape

    if bg_h < img_h or bg_w < img_w:
        raise ValueError("Background image is smaller than the input image")

    x = random.randint(0, bg_w - img_w)
    y = random.randint(0, bg_h - img_h)
    crop = bg_image[y:y + img_h, x:x + img_w]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(crop, crop, mask=mask_inv)
    digit_part = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.add(bg_part, digit_part)
    return img

def add_gaussian_noise(img):
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss * 255
    img = np.clip(noisy, 0, 255).astype(np.uint8)
    return img

def change_brightness_and_contrast(img):
    brightness = random.randint(-50, 50)
    contrast = random.randint(-30, 30)

    img = np.int16(img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img

def add_motion_blur(img):
    size = random.randint(5, 15)
    kernel_motion_blur = np.zeros((size, size))
    
    if random.choice([True, False]):
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    else:
        kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
    
    kernel_motion_blur = kernel_motion_blur / size
    img = cv2.filter2D(img, -1, kernel_motion_blur)
    return img

def custom_augmentation(img):
    # Example custom augmentation: Rotate the image by a random angle
    angle = random.uniform(-30, 30)  # Rotate between -30 and 30 degrees
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img

def combine_all_augmentations(img, bg_images):
    if random.random() < 0.5:
        img = add_gaussian_noise(img)
    if random.random() < 0.5:
        img = change_brightness_and_contrast(img)
    if random.random() < 0.5:
        img = add_motion_blur(img)
    return img

def domain_randomizations(img, bg_images, domain_randomziation_type=3):
    if domain_randomziation_type == 0:
        return img
        #print("no augmentations are applied")
    elif domain_randomziation_type == 1:
        #print("change digital color")
        img = random_digit_color(img, bg_images)
    elif domain_randomziation_type == 2:
        #print("use random background image")
        img = random_background(img, bg_images)
    elif domain_randomziation_type == 3:
        #print("add gaussian noise")
        img = random_digit_color(img, bg_images)
        img = random_background(img, bg_images)
        img = add_gaussian_noise(img)
    elif domain_randomziation_type == 4:
        #print("change brightness and contrast")
        img = random_digit_color(img, bg_images)
        img = random_background(img, bg_images)
        img = change_brightness_and_contrast(img)
    elif domain_randomziation_type == 5:
        #print("add motion blur")
        img = random_digit_color(img, bg_images)
        img = random_background(img, bg_images)
        img = add_motion_blur(img)
    elif domain_randomziation_type == 6:
        #print("use custom_augmentation")
        img = random_digit_color(img, bg_images)
        img = random_background(img, bg_images)
        img = custom_augmentation(img)
    elif domain_randomziation_type == 7:
        #print("combine all augmentations")
        img = random_digit_color(img, bg_images)
        img = random_background(img, bg_images)
        img = combine_all_augmentations(img, bg_images)
    else:
        raise AssertionError("unexpected domain_randomziation_type:", domain_randomziation_type)

    #cv2.imshow("img augmented", img)
    #cv2.waitKey(0)
    return img
