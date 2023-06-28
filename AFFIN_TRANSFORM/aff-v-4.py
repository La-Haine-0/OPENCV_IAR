import cv2
import numpy as np

# Загрузка изображений
img1 = cv2.imread('bk2.jpg')
img2 = cv2.imread('bk3.jpg')

# Преобразование изображений в градации серого
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Нахождение ключевых точек и дескрипторов с помощью детектора ORB
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Сопоставление ключевых точек
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

# Сортировка сопоставлений по расстоянию
matches = sorted(matches, key=lambda x: x.distance)

# Ограничение количества сопоставлений
num_matches = 100
matches = matches[:num_matches]

# Извлечение координат ключевых точек для сопоставления
points1 = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

# Вычисление матрицы Гомографии
H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

# Выравнивание изображений
aligned_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

# Визуализация сопоставлений и выравненного изображения
matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matched Image', matched_img)
cv2.imshow('Aligned Image', aligned_img)
cv2.imwrite("matched.jpg",matched_img)
cv2.imwrite("aligned.jpg",aligned_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
