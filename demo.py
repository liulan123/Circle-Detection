from preprocess import *

path = "image/RTS02.jpg"
image = cv.imread(path, 0)
# ShowImage(image)

edges = EdgeDetect(image)
# ShowImage(edges)
H, W = image.shape[:2]
edges = np.where(edges != 0)
edge_points = np.dstack((edges[0], edges[1])).reshape(-1, 2)

cricle_list = Hough(edge_points, H, W)
for circle in cricle_list:
    parameters, times = circle[:2]
    center_x, center_y, radius = parameters[:3]
    # print("%d %d %d" % (center_x, center_y, radius))

res = cv.imread(path, 1)
num_res = 20
for circle in cricle_list[:num_res]:
    parameters, times = circle[:2]
    center_x, center_y, radius = parameters[:3]
    center_x, center_y, radius = int(center_x), int(center_y), int(radius)
    cv.rectangle(res, (center_y - radius, center_x - radius),
                 (center_y + radius, center_x + radius), 
                 (255, 0, 0), 
                 2)

# cv.imwrite('result/RTS02_detected.jpg', res)
plt.imshow(res)
plt.show()
