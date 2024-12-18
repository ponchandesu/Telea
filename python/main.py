import sys
import numpy as np
import logging
import cv2
from queue import PriorityQueue


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def print_image(img: np.ndarray) -> None:
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            print(f"src[{i}][{j}] = {img[i][j]}")


def edit_border(img: np.ndarray) -> None:
    lst_r, lst_c = img.shape[0] - 1, img.shape[1] - 1
    img[:, 0] = 0
    img[:, lst_c] = 0
    img[0, :] = 0
    img[lst_r, :] = 0


def fms(i1: int, j1: int, i2: int, j2: int, f: np.ndarray, t: np.ndarray) -> float:
    a1, a2 = t[i1, j1], t[i2, j2]
    m = min(a1, a2)
    if f[i1, j1] != 2:
        if f[i2, j2] != 2:
            if abs(a1 - a2) >= 1.0:
                return 1.0 + m
            return 0.5 * (a1 + a2 + np.sqrt(2.0 - (a1 - a2) ** 2))
        return 1.0 + a1
    if f[i2, j2] != 2:
        return 1.0 + a2
    return 1.0 + m


def min4(a1: float, a2: float, a3: float, a4: float) -> float:
    return min(a1, a2, a3, a4)


def in_paint_point(
    i: int, j: int, f: np.ndarray, t: np.ndarray, ret: np.ndarray, epsilon: int
) -> None:
    radius_sqr = float(epsilon**2)
    grad_tx, grad_ty = 0.0, 0.0

    if f[i, j + 1] != 2:
        if f[i, j - 1] != 2:
            grad_tx = 0.5 * (t[i, j + 1] - t[i, j - 1])
        else:
            grad_tx = t[i, j + 1] - t[i, j]
    elif f[i, j - 1] != 2:
        grad_tx = t[i, j] - t[i, j - 1]

    if f[i + 1, j] != 2:
        if f[i - 1, j] != 2:
            grad_ty = 0.5 * (t[i + 1, j] - t[i - 1, j])
        else:
            grad_ty = t[i + 1, j] - t[i, j]
    elif f[i - 1, j] != 2:
        grad_ty = t[i, j] - t[i - 1, j]

    min_i, min_j = max(1, i - epsilon), max(1, j - epsilon)
    max_i, max_j = min(ret.shape[0] - 1, i + epsilon + 1), min(
        ret.shape[1] - 1, j + epsilon + 1
    )

    if len(ret.shape) == 3:
        Ia = np.zeros(3, dtype=float)
        
        s = 1.0e-20

        for k in range(min_i, max_i):
            for l in range(min_j, max_j):
                r_y, r_x = float(i - k), float(j - l)
                if (f[k, l] != 2) and ((r_x**2 + r_y**2) <= radius_sqr):
                    dst = 1.0 / ((r_x**2 + r_y**2) * np.sqrt(r_x**2 + r_y**2))
                    lev = 1.0 / (1.0 + abs(t[k, l] - t[i, j]))
                    dir_t = grad_tx * r_x + grad_ty * r_y
                    dir_t = max(dir_t, 0.000001)
                    w = abs(dst * lev * dir_t)
                    Ia += w * ret[k, l, :]
                    s += w

        ret[i, j, :] = 0.5 + Ia / s

    else:
        Ia, s = 0.0, 0.0, 0.0, 1.0e-20
        for k in range(min_i, max_i):
            for l in range(min_j, max_j):
                r_y, r_x = i - k, j - l
                if (f[k, l] != 2) and ((r_x**2 + r_y**2) <= radius_sqr):
                    dst = 1.0 / ((r_x**2 + r_y**2) * np.sqrt(r_x**2 + r_y**2))
                    lev = 1.0 / (1.0 + abs(t[k, l] - t[i, j]))
                    dir_t = grad_tx * r_x + grad_ty * r_y
                    dir_t = max(dir_t, 0.000001)
                    w = abs(dst * lev * dir_t)
                    Ia += w * ret[k, l]
                    s += w

        ret[i, j] = 0.5 + Ia / s


def telea(
    f: np.ndarray, t: np.ndarray, ret: np.ndarray, epsilon: int, heap: PriorityQueue
) -> None:
    while not heap.empty():
        val_t, (r, c) = heap.get()
        f[r, c] = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            i, j = r + dr, c + dc
            if f[i, j] == 2:
                dist = min4(
                    fms(i - 1, j, i, j - 1, f, t),
                    fms(i + 1, j, i, j - 1, f, t),
                    fms(i - 1, j, i, j + 1, f, t),
                    fms(i + 1, j, i, j + 1, f, t),
                )
                t[i, j] = dist
                in_paint_point(i, j, f, t, ret, epsilon)
                f[i, j] = 1
                heap.put((dist, (i, j)))
def inpaint_telea(
    src: np.ndarray, mask: np.ndarray, epsilon: int = 5
) -> np.ndarray:
    """
    Implements Telea's inpainting method.
    
    :param src: The source image as a NumPy array.
    :param mask: The inpainting mask as a NumPy array (non-zero values indicate the regions to be inpainted).
    :param epsilon: The radius for searching neighboring pixels.
    :return: Inpainted image as a NumPy array.
    """
    assert src.shape[:2] == mask.shape[:2], "Source and mask must have the same dimensions."
    ret = src.copy()
    f = np.zeros(mask.shape, dtype=np.uint8)
    t = np.zeros(mask.shape, dtype=np.float32)
    heap = PriorityQueue()

    # Initialize the f and t arrays
    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if mask[i, j] > 0:
                f[i, j] = 2  # Pixels to be inpainted
            else:
                f[i, j] = 0  # Known pixels

    # Find border pixels and initialize the priority queue
    

    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if f[i, j] == 2:
                if (
                    f[i - 1, j] == 0
                    or f[i + 1, j] == 0
                    or f[i, j - 1] == 0
                    or f[i, j + 1] == 0
                ):  
                    
                    f[i, j] = 1  # Mark as border pixel
                    dist = min4(
                        fms(i - 1, j, i, j - 1, f, t),
                        fms(i + 1, j, i, j - 1, f, t),
                        fms(i - 1, j, i, j + 1, f, t),
                        fms(i + 1, j, i, j + 1, f, t),
                    )
                    t[i, j] = dist
                    heap.put((dist, (i, j)))

    # Perform the inpainting
    telea(f, t, ret, epsilon, heap)
    np.savetxt("t.csv", t, delimiter=",", fmt="%.2f")
    return ret


def main():
    # Load an example image and mask
    src_path = "source.bmp"
    mask_path = "mask.png"

    src = cv2.imread(src_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if src is None or mask is None:
        logging.error("Could not load source or mask images.")
        return

    # Perform inpainting
    epsilon = 5
    inpainted_image = inpaint_telea(src, mask, epsilon)

    # Save and display the results
    output_path = "inpainted_image.png"
    cv2.imwrite(output_path, inpainted_image)
    #cv2.imshow("Inpainted Image", inpainted_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
