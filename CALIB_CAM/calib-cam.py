import opcode
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os

# Импортировать изображение
a = cv2.imread('C:/Users/User/Desktop/PY/chess/left09.jpg')
 # Выполнить преобразование цветового пространства
b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
 # Извлечь информацию о углах, указав, что количество углов в каждой строке равно 9, а количество углов в каждом столбце равно 6
ret, corners = cv2.findChessboardCorners(b, (9, 6))
 # ret - флаг, который используется для определения того, все ли угловые точки шахматной доски обнаружены.
print(ret)

# Критерии завершения итерации
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 # Извлечение субпиксельной угловой информации
corners2 = cv2.cornerSubPix(b, corners, (11, 11), (-1, -1), criteria)

# Нарисовать углы
cv2.drawChessboardCorners(a, (9, 6), corners2 ,False)
cv2.namedWindow('winname', cv2.WINDOW_NORMAL)
cv2.imshow('winname', a)
cv2.waitKey(0)

# Ядро калибровки камеры
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(opcode, imgpoints, b.shape[::-1], None, None)
 # ret - минимальное значение функции максимального правдоподобия
 # mtx - матрица внутренних параметров
 # dist - матрица параметров искажения камеры
 # rvecs для вектора вращения
 # tvecs - вектор смещения
tot_error = 0
for i in range(len(op)):
         imgpoints2, _ = cv2.projectPoints (op [i], rvecs [i], tvecs [i], mtx, dist) # обратный проект трехмерных координатных точек в пространстве
         error = cv2.norm (imgpoints [i], imgpoints2, cv2.NORM_L2) / len (imgpoints2) # средняя квадратическая ошибка (ошибка перепроецирования)
tot_error += error
print(ret, (tot_error / len(op))**0.5)
h, w = a.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix (mtx, dist, (h, w), 1) # исправить внутреннюю матрицу параметров
dst = cv2.undistort(a, mtx, dist, None)
# undistort
mapx, mapy = cv2.initUndistortRectifyMap (mtx, dist, None, newcameramtx, (w, h), 5) # используется для вычисления карты искажения
dst = cv2.remap (a, mapx, mapy, cv2.INTER_LINEAR) # применить полученную карту к изображению
np.savez("outfile", mtx, dist)
a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.subplot(121), plt.imshow(a), plt.title('source')
plt.subplot(122), plt.imshow(dst), plt.title('undistorted')
plt.show()