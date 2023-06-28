import cv2
import numpy as np

def order_points(pts):
    # Инициализация списка координат углов документа
    rect = np.zeros((4, 2), dtype="float32")

    # Сумма координат (x + y) дает наименьшее значение - верхний левый угол,
    # наибольшее значение - нижний правый угол
    s = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Разность координат (x - y) дает наименьшее значение - верхний правый угол,
    # наибольшее значение - нижний левый угол
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # Получение прямоугольной матрицы для преобразования перспективы
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Вычисление ширины и высоты нового изображения
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Создание конечных точек для нового изображения
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Преобразование перспективы
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def document_scanner(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Размытие изображения для снижения шумов
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Обнаружение границ на изображении
    edges = cv2.Canny(gray, 75, 200)

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Нахождение контура с максимальной площадью
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Получение периметра контура
    perimeter = cv2.arcLength(max_contour, True)

    # Аппроксимация контура
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # Проверка количества точек в аппроксимированном контуре
    if len(approx) == 4:
        # Преобразование перспективы
        warped = four_point_transform(image, approx.reshape(4, 2))

        # Отображение исходного и обработанного изображений
        cv2.imshow("Original", image)
        cv2.imshow("Scanned", warped)
        cv2.imwrite("orig-doc.jpg",image)
        cv2.imwrite("scan-doc.jpg",warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Не удалось обнаружить контур с 4 точками.")

# Пример использования
document_scanner("doc.jpg")