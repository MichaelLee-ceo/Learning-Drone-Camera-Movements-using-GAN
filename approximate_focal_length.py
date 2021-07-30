import math

def get_focal_length(h, w):
    height, width = h, w
    alpha = 69
    alpha_horizontal = 2 * math.atan(math.tan(alpha) * math.cos(math.atan(height/width)))
    alpha_vertical  = 2 * math.atan(math.tan(alpha) * math.sin(math.atan(height/width)))

    # print('a_H:', alpha_horizontal, '|', 'a_V:', alpha_vertical)

    d = math.sqrt(math.pow(width, 2) + math.pow(height, 2))
    fx = (d / 2) * (1 / math.tan(alpha_horizontal / 2))
    fy = (d / 2) * (1 / math.tan(alpha_vertical / 2))
    f = (width / 2) * (1 / math.tan(alpha / 2))

    # print('fx:', fx, '|', 'fy:', fy, '| f:', f)
    return f, f

