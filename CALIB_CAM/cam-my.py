import cv2
import numpy as np
import glob

# Установить параметры для поиска углов субпикселей. Используемый критерий остановки - максимальное количество циклов 30 и максимальный допуск ошибки 0,001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# Получить положение угловой точки калибровочной пластины
objp = np.zeros((4 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2)  # Построить мировую систему координат на калибровочной доске, координаты Z всех точек равны 0, поэтому просто назначаем x и y

obj_points = []  # Сохранить 3D-точки
img_points = []  # Сохранить 2D-точки

images = glob.glob("C:/Users/User/Desktop/PY/chess/*.jpg")
i = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (6, 4), None)

    if ret:
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # Найти угловые точки субпикселя на основе исходных угловых точек
        if corners2 is not None:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (6, 4), corners, ret)
        i += 1
        cv2.imwrite('conimg' + str(i) + '.jpg', img)
        cv2.waitKey(1500)

print(len(img_points))
cv2.destroyAllWindows()

# Калибровка
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx)  # матрица внутренних параметров
print("dist:\n", dist)  # коэффициенты искажения (k1, k2, p1, p2, k3)
print("rvecs:\n", rvecs)  # вектор вращения (внешние параметры)
print("tvecs:\n", tvecs)  # вектор перевода (внешние параметры)

print("-----------------------------------------------------")

img = cv2.imread(images[2])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # Отобразить больший диапазон изображений (некоторые изображения будут удалены после нормального переназначения)
print(newcameramtx)
print("------------------ Использование функции устранения искажений -------------------")
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst1 = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult3.jpg', dst1)
print("Метод 1: Размер dst:", dst1.shape)
