import numpy as np

def nearest_neighbor_interpolation(img, scale_x, scale_y):
    """
    对单通道或三通道图像进行最近邻插值缩放

    Args:
        img: 输入图像 (numpy array)
        scale_x: 宽度缩放比例 (目标宽度 / 原始宽度)
        scale_y: 高度缩放比例 (目标高度 / 原始高度)

    Returns:
        缩放后的图像 (numpy array)
    """
    # 获取原始图像尺寸和通道数
    src_h, src_w = img.shape[:2]
    channels = 1 if len(img.shape) == 2 else img.shape[2]

    # 计算目标图像尺寸
    dst_h = int(src_h * scale_y)
    dst_w = int(src_w * scale_x)

    # 创建空白目标图像
    dst_img = np.zeros((dst_h, dst_w, channels), dtype=img.dtype)

    # 遍历目标图像的每一个像素
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            # 核心：向后映射，计算对应在原图中的坐标
            # 注意：这里使用了更精确的中心对齐公式，避免图像整体偏移
            src_x = (dst_x + 0.5) * (src_w / dst_w) - 0.5
            src_y = (dst_y + 0.5) * (src_h / dst_h) - 0.5

            # 四舍五入取整，找到最近邻的整数坐标
            # 并确保坐标不超出原图边界
            src_x_int = int(round(src_x))
            src_y_int = int(round(src_y))
            src_x_int = min(max(src_x_int, 0), src_w - 1)
            src_y_int = min(max(src_y_int, 0), src_h - 1)

            # 将原图的像素值赋给目标图像
            if channels == 1:
                dst_img[dst_y, dst_x] = img[src_y_int, src_x_int]
            else:
                dst_img[dst_y, dst_x] = img[src_y_int, src_x_int, :]

    return dst_img

# --- 测试例题一的矩阵 ---
# 输入 3x3 矩阵
src_matrix = np.array([[234, 38, 22],
                       [67, 44, 12],
                       [89, 65, 63]], dtype=np.uint8)

# 放大到 4x4，即宽高缩放比例均为 4/3 ≈ 1.333
scale = 4 / 3
dst_matrix = nearest_neighbor_interpolation(src_matrix, scale, scale)

print("原始矩阵 (3x3):\n", src_matrix)
print("\n最近邻插值放大后矩阵 (4x4):\n", dst_matrix)