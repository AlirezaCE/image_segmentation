#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
img = cv2.imread('word-segmentation.JPEG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, c = img.shape

if w > 1000:

    new_w = 1000
    ar = w/h
    new_h = int(new_w/ar)

    img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)

plt.imshow(img)

# %%
def thresholding(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(thresh, cmap='gray')
    return thresh

thresh_img = thresholding(img)

# %%
kernel = np.ones((6,85), np.uint8)
dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
plt.imshow(dilated, cmap='gray')

# %%
(contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1])

# %%
img2 = img.copy()

for ctr in sorted_contours_lines:
    print(ctr)
    x, y, w, h = cv2.boundingRect(ctr)
    cv2.rectangle(img2, (x,y), (x+w, y+h), (50, 50, 50), 2)

plt.imshow(img2)

# %%
kernel2 = np.ones((10,30), np.uint8)
dilated2 = cv2.dilate(thresh_img, kernel2, iterations = 1)
plt.imshow(dilated2, cmap='gray')

# %%
img3 = img.copy()
word_list = []

for line in sorted_contours_lines:
    x, y, w, h = cv2.boundingRect(line)
    roi_line = dilated2[y:y+h, x:x+w]
    print (y)
    (contours_word, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours_words = sorted(contours_word, key = lambda ctr : cv2.boundingRect(ctr)[0])

    for word in sorted_contours_words:

        if cv2.contourArea(word) < 400:
            continue

        x2, y2, w2, h2 = cv2.boundingRect(word)
       # print(x2)
        word_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
        cv2.rectangle(img3, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (0,0,100), 2)

plt.imshow(img3)

# %%
def ShowWord(numOfWord):
    word = word_list[numOfWord]
    roi = img[word[1]:word[3], word[0]:word[2]]
    plt.imshow(roi)
    print(word)

ShowWord(19)

# %%
