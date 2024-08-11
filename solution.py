import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.spatial.distance import cdist
import cv2


def read_csv_(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs, title, ax):
    colours = ['red', 'green', 'blue', 'yellow', 'purple']  # Define some colors for plotting
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)

class RDP:
    def __init__(self, epsilon=0):
        self.epsilon = epsilon

    def pldist(self, point, start, end):
        if np.all(np.equal(start, end)):
            return np.linalg.norm(point - start)
        return np.divide(
            np.abs(np.linalg.norm(np.cross(end - start, start - point))),
            np.linalg.norm(end - start)
        )

    def _rdp_iter(self, M, start_index, last_index, epsilon, dist):
        stk = []
        stk.append([start_index, last_index])
        global_start_index = start_index
        indices = np.ones(last_index - start_index + 1, dtype=bool)

        while stk:
            start_index, last_index = stk.pop()
            dmax = 0.0
            index = start_index

            for i in range(index + 1, last_index):
                if indices[i - global_start_index]:
                    d = dist(M[i], M[start_index], M[last_index])
                    if d > dmax:
                        index = i
                        dmax = d

            if dmax > epsilon:
                stk.append([start_index, index])
                stk.append([index, last_index])
            else:
                for i in range(start_index + 1, last_index):
                    indices[i - global_start_index] = False

        return indices

    def rdp_iter(self, M, epsilon, dist, return_mask=False):
        mask = self._rdp_iter(M, 0, len(M) - 1, epsilon, dist)
        if return_mask:
            return mask
        return M[mask]

    def rdp(self, M, epsilon=None, dist=None, return_mask=False):
        if epsilon is None:
            epsilon = self.epsilon
        if dist is None:
            dist = self.pldist

        algo = partial(self.rdp_iter, return_mask=return_mask)
        if "numpy" in str(type(M)):
            return algo(M, epsilon, dist)

        return algo(np.array(M), epsilon, dist).tolist()


class GenSolveModel:
    def __init__(self, inputCSV):
        self.inputXY = read_csv_(inputCSV)
        self.outputXY = []

    def _is_polyline_a_line_segment(self, line, tolerance=1e-3) -> bool:
        if len(line) < 3:
            return True
        Lvector = line[-1] - line[0]
        normLvector = Lvector / np.linalg.norm(Lvector)
        projections = np.dot(line - line[0], normLvector)
        perp_dist = np.linalg.norm(
            (line - line[0]) - np.outer(projections, normLvector),
            axis=1
        )
        return np.all(perp_dist <= tolerance)
    
    def _join_fragmented_line_segments(self, line1, line2, distance_threshold=10):
        if not (self._is_polyline_a_line_segment(line1) and self._is_polyline_a_line_segment(line2)):
            return None

        dist_start1_start2 = np.linalg.norm(line1[0] - line2[0])
        dist_start1_end2 = np.linalg.norm(line1[0] - line2[-1])
        dist_end1_start2 = np.linalg.norm(line1[-1] - line2[0])
        dist_end1_end2 = np.linalg.norm(line1[-1] - line2[-1])

        min_dist = min(dist_start1_start2, dist_start1_end2, dist_end1_start2, dist_end1_end2)

        if min_dist > distance_threshold:
            return None

        if min_dist == dist_start1_start2:
            return np.vstack((line1[::-1], line2))
        elif min_dist == dist_start1_end2:
            return np.vstack((line1[::-1], line2[::-1]))
        elif min_dist == dist_end1_start2:
            return np.vstack((line1, line2))
        else:  # min_dist == dist_end1_end2
            return np.vstack((line1, line2[::-1]))

    def _straigthen_colinear_line_segments(self, line1, line2, tolerance=1e-3):
        joined_line = self._join_fragmented_line_segments(line1, line2)
        if joined_line is None:
            return None
        if self._is_polyline_a_line_segment(joined_line, tolerance):
            return np.array([joined_line[0], joined_line[-1]])
        return None

    def _isFixableConic(self, polyline):
        def calculate_fitting_error(polyline, shape):
            (cx, cy), (w, h), angle = shape
            error = 0
            for point in polyline:
                px, py = point
                if w == h:  # Circle
                    distance = np.sqrt((px - cx)**2 + (py - cy)**2)
                    radius = w / 2
                    error += np.abs(distance - radius)
                else:  # Ellipse
                    cos_angle = np.cos(np.radians(-angle))
                    sin_angle = np.sin(np.radians(-angle))
                    x = (px - cx) * cos_angle - (py - cy) * sin_angle
                    y = (px - cx) * sin_angle + (py - cy) * cos_angle
                    ellipse_radius = np.sqrt((x**2 / (w / 2)**2) + (y**2 / (h / 2)**2))
                    error += np.abs(ellipse_radius - 1)
            return error / len(polyline)

        points = np.array(polyline).reshape(-1, 1, 2).astype(np.float32)
        if len(points) >= 5:
            ellipse = cv2.fitEllipse(points)
            circle = cv2.minEnclosingCircle(points)

            ellipse_error = calculate_fitting_error(polyline, ellipse)
            circle_error = calculate_fitting_error(polyline, (circle[0], (circle[1]*2, circle[1]*2), 0))
            print(circle_error, ellipse_error)

            if circle_error <= 5 and np.linalg.norm(polyline[0] - polyline[-1]) < 2:
                return "Circle", [circle[0], circle[1]]
            elif ellipse_error < 0.01 and np.linalg.norm(polyline[0] - polyline[-1]) < 2:
                return "Ellipse", [ellipse[0][0], ellipse[0][1], ellipse[1][0]/2, ellipse[1][1]/2]

        return None, None

    def _isFixablePolygon(self, polyline):
        if self._is_polyline_a_line_segment(polyline, tolerance=1e-3):
            return 1, np.array([polyline[0], polyline[-1]])

        return None, None

    def getPathXY(self):
        for XYs in self.inputXY:
            reduced_XYs = []
            rdp_instance = RDP(epsilon=10)
            for XY in XYs:
                shape_id, props = self._isFixableConic(XY)
                if shape_id is not None:
                    if shape_id == "Circle":
                        center, radius = props
                        t = np.linspace(0, 2*np.pi, 100)
                        circle = np.column_stack((center[0] + radius*np.cos(t), center[1] + radius*np.sin(t)))
                        reduced_XYs.append(circle)
                    elif shape_id == "Ellipse":
                        c1, c2, r1, r2 = props
                        t = np.linspace(0, 2*np.pi, 100)
                        ellipse = np.column_stack((c1 + r1*np.cos(t), c2 + r2*np.sin(t)))
                        reduced_XYs.append(ellipse)
                else:
                    rdpXY = rdp_instance.rdp(XY)
                    shape_id, props = self._isFixablePolygon(rdpXY)
                    if shape_id is not None:
                        if shape_id == 1:  # Line
                            reduced_XYs.append(props)
                    else:
                        reduced_XYs.append(rdpXY)
            
            i = 0
            while i < len(reduced_XYs) - 1:
                straightened = self._straigthen_colinear_line_segments(reduced_XYs[i], reduced_XYs[i+1])
                if straightened is not None:
                    reduced_XYs[i] = straightened
                    reduced_XYs.pop(i+1)
                else:
                    i += 1
            
            self.outputXY.append(reduced_XYs)
        return self.outputXY

# Read and process the CSV files
inputcsv = "./problems/frag2.csv"
outputcsv = "./problems/frag2_sol.csv"

    
inputXY = read_csv_(inputcsv)
outputXY = read_csv_(outputcsv)

resultXYModel = GenSolveModel(inputcsv)
resultXY = resultXYModel.getPathXY()

# Plot side by side
fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(24, 8))
plot(inputXY, 'Input Data', axs[0])
plot(resultXY, 'Result', axs[1])
plot(outputXY, 'Expected Result', axs[2])
plt.show()
