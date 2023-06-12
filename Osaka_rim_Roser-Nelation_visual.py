import os
import math
from pickle import TRUE
import vtk
from utils import vis_utils
from utils.VTK import VTK_Helper
import logging
from math import atan2, degrees
import pandas as pd
import tqdm
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


_CASE_LIST = '//Salmon/User/Chen/Vessel_data/Dataset for Distance Assessment/crop_vessels_pelvis_36cases/caseid_list_vessels_36.txt'
_GT_POLY_ROOT ='//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons'

_Rim_ROOT = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/acetablum_points'

_Pelvis_ROOT = _GT_POLY_ROOT

_ASIS_ROOT =  '//Salmon/User/Chen/Vessel_data/20230222_ASIS_Landmarks_Osaka.csv'
_ASIS_landmark = pd.read_csv(_ASIS_ROOT, header=0, index_col=0)

_IT_ROOT =  '//Salmon/User/Chen/Vessel_data/20230226_IT_Landmarks_Osaka.csv'
_IT_landmark = pd.read_csv(_IT_ROOT, header=0, index_col=0)
_FIG_TGT = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/20230328/PolyFigures_CEll'
_Dis_TGT = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/20230328/PolyDistances'
_Dis_TGT_vessel = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/20230328/PolyDistances_Vessel'
_Dis_TGT_index = '//Salmon/User/mazen/Segmentation/Codes/results/Osaka/vessels/36_womuscles_cropped_150000/seperation_left_right/Polygons/20230328/PolyDistances_pelvis_id'
os.makedirs(_FIG_TGT, exist_ok=True)
os.makedirs(_Dis_TGT, exist_ok=True)
os.makedirs(_Dis_TGT_vessel, exist_ok=True)
_DIST_MINMAX = (0, 20)
# AZs = [65, 95, 125]

AZs = [30,60,90,120]
_SUB_ALPHAS  =[0.3,
                0.2,
               0.9,
              0.6]
def write_point_to_file(f, p):
    with open(f, 'w') as fw:
        fw.write(f'{p[0]}, {p[1]}, {p[2]}\n')
import numpy as np

def find_intersection_points(points, center, vector):
    # 建立以veter_line为X轴的坐标系
    x_axis = vector / np.linalg.norm(vector)
    z_axis = np.cross(x_axis, np.array([0, 0, 1]))
    y_axis = np.cross(z_axis, x_axis)

    # 将点变换到该坐标系下
    points_transformed = []
    for p in points:
        p_transformed = np.array([p[0], p[1], p[2]]) - center
        p_transformed = np.dot(p_transformed, np.array([x_axis, y_axis, z_axis]))
        points_transformed.append(p_transformed)

    # 找到每个点在XOY平面上的投影点
    points_projected = [(p[0], p[1], 0) for p in points_transformed]

    # 找到X轴和Y轴上的点
    point_x = np.array([1, 0, 0])
    point_y = np.array([0, 1, 0])

    # 计算X轴和Y轴与每个点的连线的交点
    intersection_points = []
    for p in points_projected:
        # 计算X轴上的交点
        t_x = -p[0] / point_x[0]
        intersection_x = p + t_x * point_x

        # 计算Y轴上的交点
        t_y = -p[1] / point_y[1]
        intersection_y = p + t_y * point_y

        # 把两个交点的坐标存到结果列表中
        intersection_points.append((intersection_x[0], intersection_x[1], intersection_x[2]))
        intersection_points.append((intersection_y[0], intersection_y[1], intersection_y[2]))

    return intersection_points


def rotate_point_around_axis(point, axis_point, axis_direction, angle):
    # Calculate the vector from the axis point to the point
    v = np.array(point) - np.array(axis_point)

    # Calculate the Rodrigues rotation formula
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    k = np.array(axis_direction)
    kv = np.cross(k, v)
    rotated_point = point + cos_theta * v + sin_theta * kv + (1 - cos_theta) * np.dot(k, v) * k

    return rotated_point


def compute_angle(point, center, vector):
    # 计算相对于中心的偏移量
    x = point[0] - center[0]
    y = point[1] - center[1]
    z = point[2] - center[2]

    # 将向量旋转到X轴上
    x_axis = np.array([1, 0, 0])
    rotation_axis = np.cross(vector, x_axis)
    rotation_angle = np.arccos(np.dot(vector, x_axis) / (np.linalg.norm(vector) * np.linalg.norm(x_axis)))
    rotated_point = rotate_point_around_axis(point, center, rotation_axis, rotation_angle)

    # 计算旋转后的点相对于中心的偏移量
    rx = rotated_point[0] - center[0]
    ry = rotated_point[1] - center[1]
    rz = rotated_point[2] - center[2]

    # 计算角度并将其转换为度数
    angle = atan2(ry, rx)
    angle_degrees = degrees(angle)
    return angle_degrees


def angle_with_x_axis(point, x_rosar, center):
    # 将点和 x_rosar 向量转换为 numpy 数组
    point_array = np.array(point)
    x_rosar_array = np.array(x_rosar)

    # 将点和 x_rosar 向量的坐标都减去 center 的坐标
    shifted_point = point_array - np.array(center)
    shifted_x_rosar = x_rosar_array - np.array(center)

    # 将 x_rosar 向量调整为 X 轴的方向
    theta = np.arctan2(shifted_x_rosar[1], shifted_x_rosar[0])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta, cos_theta, 0],
                  [0, 0, 1]])
    rotated_x_rosar = np.dot(R, shifted_x_rosar)

    # Calculate the angle between the shifted point and rotated x_rosar vectors
    angle = math.degrees(math.acos(
        np.dot(shifted_point, rotated_x_rosar) / (np.linalg.norm(shifted_point) * np.linalg.norm(rotated_x_rosar))))

    return angle



def point2point_distance(p1, p2):
    distance = ((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2 + (
                float(p1[2]) - float(p2[2])) ** 2) ** 0.5
    return distance

def read_datalist(fpath, root = None, ext=None,field='fileID'):
    datalist = []
    if fpath.endswith('.txt'):
        datalist = np.genfromtxt(fpath, dtype=str)
    elif fpath.endswith('.csv'):
        df = pd.read_csv(fpath,header=0)
        # print('Dataframe: ', df)
        datalist = df[field].values.tolist()
    if root:
        datalist = ['%s/%s' % (root,item) for item in datalist]
    if ext:
        datalist = ['%s/%s' % (item,ext) for item in datalist]
    print(datalist)
    return datalist
def signed_angle_between_vectors(v1, v2):
    """
    Returns the signed angle in degrees between two 3D vectors, using the normal vector
    between them to determine the sign of the angle and the direction of the rotation axis.
    """
    # Calculate the dot product and the cross product
    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)

    # Calculate the magnitude of the cross product
    magnitude_cross = np.linalg.norm(cross_product)

    # Calculate the normal vector between the two input vectors
    normal = cross_product / magnitude_cross

    # Calculate the angle in radians between the two vectors
    angle_rad = np.arctan2(magnitude_cross, dot_product)

    # Determine the sign of the angle using the normal vector
    angle_sign = np.sign(np.dot(v1, np.cross(v2, normal)))

    # Convert the angle to degrees and return the signed value
    angle_deg = np.degrees(angle_rad) * angle_sign
    return angle_deg

def calculate_angles(point, center, roser_vec):
    # Calculate the angle between roser_vec and x axis
    angle_roser_x = signed_angle_between_vectors(roser_vec, np.array([1, 0, 0]))

    # Translate all points so that center is the origin
    translated_point = point - center

    # Rotate all points around z axis by -angle_roser_x degrees
    rotation_matrix = np.array([[np.cos(angle_roser_x), -np.sin(angle_roser_x), 0],
                                [np.sin(angle_roser_x), np.cos(angle_roser_x), 0],
                                [0, 0, 1]])
    rotated_point = np.dot(translated_point, rotation_matrix)

    # Calculate the angle between each point's vector and x axis
    point_vec = np.array(rotated_point)
    angle = signed_angle_between_vectors(point_vec, np.array([1, 0, 0]))
    if point[1] < 0:
        angle *= -1

    return angle

def get_right_quadrant(point, center, roser_vec):
    point_vec = np.array(point) - np.array(center)
    dot_product = np.dot(point_vec, roser_vec)
    if dot_product > 0:
        if point_vec[0] > 0:
            return 1
        else:
            return 4
    elif dot_product < 0:
        if point_vec[0] > 0:
            return 2
        else:
            return 2


def get_left_quadrant(point, center, roser_vec):
    point_vec = np.array(point) - np.array(center)
    dot_product = np.dot(point_vec, roser_vec)
    if dot_product > 0:
        if point_vec[0] > 0:
            return 1
        else:
            return 4
    elif dot_product < 0:
        if point_vec[0] > 0:
            return 2
        else:
             return 2
    # # if point_vec[1] <= 0 else 3
    # else:
    #     if point_vec[0] > 0:
    #         return 2 if point_vec[1] <= 0 else 3
    #     else:
    #         return 2 if point_vec[1] <= 0 else 3

def VisualizeSurfaceDistance(
        src_poly_path,
        tgt_poly_path,
        rim_poly_path,
        _asis_path,
        _it_path,
        case,
        az,
        sub_tgt_poly_path=[],
        sub_tgt_poly_colors=[],
        sub_tgt_poly_alphas=[],
        tag = ''):
    '''
    Function to generate polygon data from label file.
    Inputs:
        in_label_path: location of the input label file
        out_file_path: location of output polygon file
        tgt_num_cells: number of cells to be generated
        label_val    : target label number (integer between 1 and max(labels))
        out_ext      : extension of the polygon data file
        vis          : flag indicating whether to visualize the generated polygon
        tag          : tag to add to the snapshot file name
    Outputs:
        None
    '''
    try:
        #Read label image
        logging.info('Generation settings \n %s\n' % (locals()))
        reader_src = vis_utils.get_poly_reader(_SRC)
        reader_tgt = vis_utils.get_poly_reader(_TGT)




        logging.info('Surface (polygon) generated')

        # Surface reaser
        num_cells_src = reader_src.GetOutput().GetNumberOfCells()
        num_cells_tgt = reader_tgt.GetOutput().GetNumberOfCells()
        logging.info('Current number of cells inm source: %d' % (num_cells_src))
        logging.info('Current number of cells inm source: %d' % (num_cells_tgt))

        if 'vein' in _SRC_STRUCT:
            src_actor = vis_utils.get_poly_actor(
                    reader_src,
                    edge_visible=False,
                    col=(0.0, 0.6, 0.8),  # vein
                    alpha=1
                    )
        elif'artery' in _SRC_STRUCT:
            src_actor = vis_utils.get_poly_actor(
                reader_src,
                edge_visible=False,
                col=(0.8, 0.45, 0.25),  # artery

                alpha=1
            )


        _sub_actors = []
        if _SUB_ACTORS is not None:
            for _sub, _col, _alpha in zip(sub_tgt_poly_path,
                                          sub_tgt_poly_colors,
                                          sub_tgt_poly_alphas):
                _tmp_reader = vis_utils.get_poly_reader(_sub)
                _sub_actors.append(vis_utils.get_poly_actor(
                    _tmp_reader,
                    edge_visible=False,
                    col=_col,
                    alpha=_alpha
                    ))
        logging.info('Actors loaded')
        # renderer.AddActor(src_actor)
        # renderer.AddActor(tgt_actor)
        # c = src_actor.GetOutput().GetCenter()
        # pos = [600, 0, 0]
        logging.info('Camera position set')

        logging.info('Renderer set')
        # Show
        renderer, renWindow = vis_utils.get_poly_renderer(bg = (1.0, 1.0, 1.0),
                                                          off_screen=True,
                                                          gradient_bg = TRUE)
        # logging.info('Renderer and window loaded')
        _tmp_reader = vis_utils.get_poly_reader(_TGT)
        c = _tmp_reader.GetOutput().GetCenter()
        if 'nerve' in _SRC_STRUCT:
            if _side == 'right':
                pos = [c[0] ,
                       c[1] - 350,
                       c[2] ]
            elif _side == 'left':
                pos = [c[0] ,
                       c[1] - 350,
                       c[2] ]
            print(pos)
            renderer = vis_utils.set_renderer_camera(renderer,
                                                     pos=pos,
                                                     fc=c,
                                                     el=0,
                                                     # az=-120,pos = [c[0]+250,
                                                     #                    c[1] - 500,
                                                     #                    c[2]+250]
                                                     az= -95 if _side == 'right' else 95,
                                                     roll=0)
        elif 'artery' or 'vein' in _SRC_STRUCT:
            pos = [c[0]+0,
                   c[1] + 500,
                   c[2] +0]
            print(pos)
            print (az)
            renderer = vis_utils.set_renderer_camera(renderer,
                                                     pos=pos,
                                                     fc=c,
                                                     el=0,
                                                     # az=-120,
                                                     az= az if _side == 'right' else -az,
                                                     roll=0)
        renWindow.Render()

        # read landmark
        logging.info('Started read landmarks')
        _ASIS_landmark = _asis_path
        # a=_ASIS_landmark.loc[case.capitalize()]['ASIS_rt_x']
        if _side == 'right' :
            _ASIS_point = [-_ASIS_landmark.loc[case.capitalize()]['ASIS_rt_x'], -_ASIS_landmark.loc[case.capitalize()]['ASIS_rt_y'],
                            _ASIS_landmark.loc[case.capitalize()]['ASIS_rt_z']]
        elif _side == 'left' :
            _ASIS_point = [-_ASIS_landmark.loc[case.capitalize()]['ASIS_lt_x'], -_ASIS_landmark.loc[case.capitalize()]['ASIS_lt_y'],
                            _ASIS_landmark.loc[case.capitalize()]['ASIS_lt_z']]

        print ("_ASIS_point:",_ASIS_point )

        _IT_landmark = _it_path
        # a=_ASIS_landmark.loc[case.capitalize()]['ASIS_rt_x']
        if _side == 'right':
            _IT_point = [-_IT_landmark.loc[case.capitalize()]['it_rt_x'],
                           -_IT_landmark.loc[case.capitalize()]['it_rt_y'],
                           _IT_landmark.loc[case.capitalize()]['it_rt_z']]
        elif _side == 'left':
            _IT_point = [-_IT_landmark.loc[case.capitalize()]['it_lt_x'],
                           -_IT_landmark.loc[case.capitalize()]['it_lt_y'],
                           _IT_landmark.loc[case.capitalize()]['it_lt_z']]

        print("_IT_point:", _IT_point)
        end_sphere_source = vtk.vtkSphereSource()
        end_sphere_source.SetCenter(_ASIS_point)
        end_sphere_source.SetRadius(5)
        end_sphere_mapper = vtk.vtkPolyDataMapper()
        end_sphere_mapper.SetInputConnection(end_sphere_source.GetOutputPort())
        end_sphere_actor = vtk.vtkActor()
        end_sphere_actor.SetMapper(end_sphere_mapper)
        end_sphere_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        renderer.AddActor(end_sphere_actor)

        logging.info('Started rendering')
        pelvis_reader = vtk.vtkPolyDataReader()
        pelvis_reader.SetFileName(_TGT)
        pelvis_reader.Update()
        pelvis_polydata = pelvis_reader.GetOutput()

        min_z_point = _IT_point
        end_sphere_source = vtk.vtkSphereSource()
        end_sphere_source.SetCenter(min_z_point)
        end_sphere_source.SetRadius(5)
        end_sphere_mapper = vtk.vtkPolyDataMapper()
        end_sphere_mapper.SetInputConnection(end_sphere_source.GetOutputPort())
        end_sphere_actor = vtk.vtkActor()
        end_sphere_actor.SetMapper(end_sphere_mapper)
        end_sphere_actor.GetProperty().SetColor(1.0, 0.0, 1.0)
        renderer.AddActor(end_sphere_actor)

        print("Min z point:", min_z_point)

        roser_source = vtk.vtkLineSource()
        roser_source.SetPoint1(_ASIS_point)
        roser_source.SetPoint2(min_z_point)
        roser_mapper = vtk.vtkPolyDataMapper()
        roser_mapper.SetInputConnection(roser_source.GetOutputPort())
        roser_actor = vtk.vtkActor()
        roser_actor.SetMapper(roser_mapper)
        roser_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        renderer.AddActor(roser_actor)

        _RIM_reader = vtk.vtkXMLPolyDataReader()


        # Set the file name for the reader
        _RIM_reader.SetFileName(_RIM)

        # Read the file
        _RIM_reader.Update()
        # 从_RIM_reader中获取所有的点并计算中心点
        rim_points = _RIM_reader.GetOutput().GetPoints().GetData()
        center = [0, 0, 0]
        for i in range(rim_points.GetNumberOfTuples()):
            point = rim_points.GetTuple3(i)
            center[0] += point[0]
            center[1] += point[1]
            center[2] += point[2]
        center = [coord / rim_points.GetNumberOfTuples() for coord in center]

        # 构建Rosser线段并计算方向向量
        roser_vec = [min_z_point[i] - _ASIS_point[i] for i in range(3)]
        roser_length = sum([coord ** 2 for coord in roser_vec]) ** 0.5
        roser_vec = [coord / roser_length for coord in roser_vec]

        line_start = [center[i] - 100 * roser_vec[i] for i in
                    range(3)]

        # Define the end point of the line segment by adding the direction vector scaled by some distance
        line_end = [center[i] + 100 * roser_vec[i] for i in
                    range(3)]  # the 100 here is an example distance, you can adjust it as needed


        # Create a vtkLineSource object and set its start and end points
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(line_start)
        line_source.SetPoint2(line_end)
        line_source.Update()
        start_end_line_mapper = vtk.vtkPolyDataMapper()
        start_end_line_mapper.SetInputConnection(line_source.GetOutputPort())
        start_end_line_actor = vtk.vtkActor()
        start_end_line_actor.SetMapper(start_end_line_mapper)
        start_end_line_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        renderer.AddActor(start_end_line_actor)

        # Define a vector that is perpendicular to the roser_vec (we can use the cross product with [1, 0, 0] for simplicity)
        perpendicular_vec = np.cross(roser_vec, [1, 0, 0])
        perpendicular_vec /= np.linalg.norm(perpendicular_vec)  # normalize the vector

        # Define the start and end points of the perpendicular line segment
        perpendicular_length = 100  # the length of the line segment (you can adjust this as needed)
        perpendicular_start = [center[i] - perpendicular_length * perpendicular_vec[i] for i in range(3)]
        perpendicular_end = [center[i] +  perpendicular_length * perpendicular_vec[i] for i in range(3)]

        x_vector = [center[i] - perpendicular_start[i] for i in range(3)]

        # Create a vtkLineSource object for the perpendicular line segment and set its start and end points
        perpendicular_source = vtk.vtkLineSource()
        perpendicular_source.SetPoint1(perpendicular_start)
        perpendicular_source.SetPoint2(perpendicular_end)

        # Update the perpendicular source and get its output
        perpendicular_source.Update()
        start_end_line_mapper = vtk.vtkPolyDataMapper()
        start_end_line_mapper.SetInputConnection(perpendicular_source.GetOutputPort())
        start_end_line_actor = vtk.vtkActor()
        start_end_line_actor.SetMapper(start_end_line_mapper)
        start_end_line_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        renderer.AddActor(start_end_line_actor)

        # rim_points = VTK_Helper.vtk_to_numpy(rim_points)
        # intersection_points = find_intersection_points(rim_points, center, x_vector)

        start_sphere_source = vtk.vtkSphereSource()
        start_sphere_source.SetCenter(line_start)
        start_sphere_source.SetRadius(5)
        start_sphere_mapper = vtk.vtkPolyDataMapper()
        start_sphere_mapper.SetInputConnection(start_sphere_source.GetOutputPort())
        start_sphere_actor = vtk.vtkActor()
        start_sphere_actor.SetMapper(start_sphere_mapper)
        start_sphere_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        # renderer.AddActor(start_sphere_actor)

        end_sphere_source = vtk.vtkSphereSource()
        end_sphere_source.SetCenter(line_end)
        end_sphere_source.SetRadius(5)
        end_sphere_mapper = vtk.vtkPolyDataMapper()
        end_sphere_mapper.SetInputConnection(end_sphere_source.GetOutputPort())
        end_sphere_actor = vtk.vtkActor()
        end_sphere_actor.SetMapper(end_sphere_mapper)
        end_sphere_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        # renderer.AddActor(end_sphere_actor)

        center_sphere_source = vtk.vtkSphereSource()
        center_sphere_source.SetCenter(center)
        center_sphere_source.SetRadius(5)
        center_sphere_mapper = vtk.vtkPolyDataMapper()
        center_sphere_mapper.SetInputConnection(center_sphere_source.GetOutputPort())
        center_sphere_actor = vtk.vtkActor()
        center_sphere_actor.SetMapper(center_sphere_mapper)
        center_sphere_actor.GetProperty().SetColor(0.0, 0.0, 1.0)
        renderer.AddActor(center_sphere_actor)

        # Calculate the quadrants for each rim point with respect to the perpendicular line segment
        # quadrants = []
        rim_points = VTK_Helper.vtk_to_numpy(rim_points)
        # quadrants = split_third_quadrant(points=rim_points,center=center,roser_vec=roser_vec)

        # Calculate polar and longitude axes
        import math

        # Define center and vector

        vector = [center[i] - line_start[i] for i in range(3)]

        # Calculate polar and longitude axes
        magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
        longitude_axis = (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)
        # longitude_axis = (-polar_axis[1], polar_axis[0], 0)

        # Calculate spherical coordinates of a point
        def spherical_coordinates(point):
            x, y, z = center[0]-point[0], center[1]-point[1], center[2]-point[2]
            r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            theta = math.acos(z / r) * 180 / math.pi
            # Calculate phi using the longitude axis
            # Calculate phi using the longitude axis
            phi = math.atan2(y * longitude_axis[0] - x * longitude_axis[1],
                             x * longitude_axis[0] + y * longitude_axis[1]) * 180 / math.pi
            if phi < 0:
                phi += 360

            return r, theta, phi

        r, theta, phi = spherical_coordinates(line_start)
        start_phi = phi


        # Divide a point into four regions
        def divide_right_point(point):
            r, theta, phi = spherical_coordinates(point)
            if phi >  0 and phi <  90:
                region = "high-risk"
            elif phi > 315 and phi < 360:
                region = "high-risk"
            elif phi < 180 and theta > 90:
                region = "superir"
            elif phi < 180 and theta <= 90:
                region = "anter"
            else:
                region = "down"
            if phi < 90:
                left_right = "right"
            else:
                left_right = "left"
            if region == "superir":
                if left_right == "left":
                    region += "-left"
                else:
                    region += "-right"
            return region

        def divide_left_point(point):
            r, theta, phi = spherical_coordinates(point)
            if phi > 0  and phi < 45:
                region = "high-risk"
            elif phi >= 270 and theta <=360:
                region = "high-risk"
            elif phi < 180 and theta <= 90:
                region = "anter"
            else:
                region = "down"
            if phi < 90:
                left_right = "right"
            else:
                left_right = "left"
            if region == "superir":
                if left_right == "left":
                    region += "-left"
                else:
                    region += "-right"
            return region


        regions = []
        quadrants =[]
        angles = []
        for point in rim_points:
            r, theta, phi = spherical_coordinates(point)
            if _side == 'right':
              region = divide_right_point(point=point)
            elif _side == 'left':
              region = divide_left_point(point=point)
            regions.append(region)
            angles.append(phi)
            quadrant = get_right_quadrant(point, center, vector)
            quadrants.append(quadrant)
        print(regions)

        quadrants.append(1)








        # Get the points from the reader's output
        points = _RIM_reader.GetOutput().GetPoints()

        # Print the points
        MIN_Distance = []
        Closest_points = []
        Pelvis_index = []
        angles = []
        for i in range(0, points.GetNumberOfPoints(), 1):
            point = points.GetPoint(i)
            point_vec = np.array(center) - np.array(point)
            # point_vec = np.array(point) - np.array(center)
            #
            # # Calculate the dot product between the point vector and the perpendicular vector
            # dot_product = np.dot(point_vec, roser_vec)

            # Determine which quadrant the current rim point belongs to based on the sign of the dot product
            # roser_vec = x_vector
            # point_vec = point - center
            # shifted_point = [point[i] - center[i] for i in range(3)]
            angle = compute_angle(point, center, x_vector)
            angles.append(angle)
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(_SRC)
            reader.Update()
            # Get the 3D model from the reader
            model = reader.GetOutput()
            pelvis_model = pelvis_reader.GetOutput()

            # Create a vtkPolyData.PointLocator object and initialize it with the model data
            pointLocator = vtk.vtkPointLocator()
            pointLocator.SetDataSet(model)
            pointLocator.BuildLocator()

            pelvis_pointLocator = vtk.vtkPointLocator()
            pelvis_pointLocator.SetDataSet(pelvis_model)
            pelvis_pointLocator.BuildLocator()

            # Use the pointLocator object to find the minimum distance between the point and the model
            minDistance_id = pointLocator.FindClosestPoint(point)
            Pelvis_closet_id = pelvis_pointLocator.FindClosestPoint(point)
            Pelvis_index.append(Pelvis_closet_id)
            points_rim_array_npy = VTK_Helper.vtk_to_numpy(model.GetPoints().GetData())
            closestPoint = points_rim_array_npy[minDistance_id]
            Closest_points.append(closestPoint)
            min_distance = point2point_distance(point, closestPoint)
            MIN_Distance.append(min_distance)
            # print(min_distance)
            # Create a vtkLineSource object to draw the minimum distance line
            line = vtk.vtkLineSource()
            line.SetPoint1(point)
            line.SetPoint2(closestPoint)



                #   sphereActor.GetProperty().SetColor(0.0, 1.0, 0.0)

                # Add the sphereActor to the scene
                # renderer.AddActor(sphereActor)

            textActor = vtk.vtkTextActor3D()
            textActor.SetInput("{:.3f}mm".format(min_distance))
            midpoint = closestPoint
            # Set the position of the text actor to the midpoint of the line
            textActor.SetPosition(midpoint[0], midpoint[1], midpoint[2])

        num_cells_tgt = reader_tgt.GetOutput().GetNumberOfCells()
        cell_data = np.ones(num_cells_tgt) * 7

        pelvis_patch = vtk.vtkPolyData()

        # Create an array to store the colors of the points in the pelvis patch
        cell_colors = vtk.vtkUnsignedCharArray()
        cell_colors.SetNumberOfComponents(3)
        cell_colors.SetName("Colors")

        ## Loop through the closest pelvis point IDs ###
        for i in range(len(Pelvis_index)):
            # Get the closest pelvis point ID
            closest_pelvis_point_id = Pelvis_index[i]
            min_distance = MIN_Distance[i]
            angle = angles[i]
            region = regions [i]
            quadrant = quadrants[i]

            # Get the coordinates of the closest pelvis point
            closest_pelvis_point = pelvis_model.GetPoint(closest_pelvis_point_id)

            # Get the cell IDs that contain the closest pelvis point
            cell_ids = vtk.vtkIdList()
            pelvis_model.GetPointCells(closest_pelvis_point_id, cell_ids)
            a = cell_ids.GetNumberOfIds()

            # Loop through the cell IDs
            for j in range(cell_ids.GetNumberOfIds()):
                # Get the cell ID
                cell_id = cell_ids.GetId(j)
                if region=='high-risk':
                  if min_distance < 10.0:
                    cell_data[cell_id] = 0
                  if min_distance < 15.0:
                    cell_data[cell_id] = 1
                  elif min_distance < 20.0:
                    cell_data[cell_id] = 2
                  elif min_distance < 25.0:
                    cell_data[cell_id] = 3
                  elif min_distance < 30.0:
                    cell_data[cell_id] = 4
                  elif min_distance < 35.0:
                    cell_data[cell_id] = 5
                  elif min_distance < 40.0:
                    cell_data[cell_id] = 6
                  else:
                      cell_data[cell_id] = 6

                else:
                    cell_data[cell_id] = 7

            # Add the cell colors array to the pelvis patch data
        reader_tgt.GetOutput().GetCellData().SetScalars(VTK_Helper.numpy_to_vtk(cell_data))
        # print(reader_tgt.GetOutput().GetCellData().SetScalars())

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(8)
        lut.SetTableRange(0, 7)
        lut.SetTableValue(0, 1.0, 0.0, 0.0, 1.0)    # Red
        lut.SetTableValue(1, 1.00000, 0.40000, 0.00000, 1.0)  # Maximum Yellow Red
        lut.SetTableValue(2,   1.00000, 0.80000, 0.00000,1.0)  # Naples Yellow
        lut.SetTableValue(3,  0.80000, 1.00000, 0.40000, 1.0)    #   Aquamarine
        lut.SetTableValue(4, 0.00000, 1.00000, 0.00000, 1.0)  #cyan
        lut.SetTableValue(5,  0.00000, 0.80000, 1.00000, 1.0) #green
        lut.SetTableValue(6, 0.00000, 0.00000, 0.60000, 1.0)
        lut.SetTableValue(7, 1.0, 1.0, 1.0, 0.2)
        lut.Build()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader_tgt.GetOutputPort())
        mapper.SetScalarRange(0, 8)  # Maximum value of the scalar (distance)
        mapper.SetLookupTable(lut)
        mapper.SetScalarModeToUseCellData()
        mapper.Update()
        # print(reader_tgt.GetOutput().GetCellData().SetScalars())

        actor = vis_utils.get_poly_actor(reader_tgt,
                                             edge_visible=False,
                                             col=(0.9, 0.9, 0.9),
                                             alpha=1)
        actor.SetMapper(mapper)
        renderer.AddActor(actor)


        # add min distance text
        MIN = MIN_Distance
        # closest_vessel_point = Closest_points[MIN.index(min(MIN))]
        point_vessel_actor = vis_utils.get_sphere_actor(Closest_points[MIN.index(min(MIN))])
        min_dis = pd.DataFrame({'min_distance': MIN_Distance, 'angle': angles, 'area': regions})
        min_dis.to_csv(os.path.join(_Dis_TGT, '%s_%s.csv' % (case, _SRC_STRUCT)), index=False)
        Closest_points = np.array(Closest_points)
        np.savetxt(os.path.join(_Dis_TGT_vessel, '%s_%s.csv' % (case, _SRC_STRUCT)), Closest_points, fmt='%0.4f',
                   delimiter=",")
        # Pelvis_index = np.array(Pelvis_index)
        # np.savetxt(os.path.join(_Dis_TGT_index, '%s_%s.csv' % (case, _SRC_STRUCT)), Pelvis_index, fmt='%d',
        #            delimiter=",")

        min_dist_vessel = np.min(MIN_Distance)
        fig_txt = 'Min distance: %0.3f mm' % (min_dist_vessel)
        txt_actor = vis_utils.get_text_actor(fig_txt, font_size=24)
        # add color bar
        # dist_actor, dist_mapper = vis_utils.get_distance_actor(dfilter_vessel,
        #                                                        minmax=_DIST_MINMAX)  #
        # rim_dist_actor, _ = vis_utils.get_distance_actor(dfilter_rim,
        #                                                  minmax=_DIST_MINMAX)
        logging.info('Distance actor initialized')
        logging.info('Point actor initialized')
        # Assign actor to the renderer.
        renderer.AddActor(src_actor)

        # renderer.AddActor(tgt_actor)
        # renderer.AddActor(dist_actor)
        # # renderer.AddActor(rim_dist_actor)
        renderer.AddActor(point_vessel_actor)
        # # renderer.AddActor(point_rim_actor)
        # renderer.AddActor(bar_actor)

        renderer.AddActor(txt_actor)

        if _SUB_ACTORS is not None:
            for _rd in _sub_actors:
                renderer.AddActor(_rd)

        fname = os.path.join(_FIG_TGT, os.path.basename(_SRC).replace('.vtk', '_' + str(az) + '_rim.png'))
        vis_utils.save_snapshot(renWindow, fname)
        del renWindow
        return min_dist_vessel


    except Exception as exc:
        logging.info('%s' % exc)
        logging.info('Files \n%s \n%s\nnot generated...' % (src_poly_path, tgt_poly_path))


if __name__ == '__main__':
    _cases = read_datalist(_CASE_LIST)
    # _cases = ['k8041']
    for _organ1, _organ2 in zip(('artery', 'vein'), ('vein',  'artery')):
        _sides = ['left', 'right']
        for i, _side in enumerate(['right', 'left']):
            # _TGT_STRUCT = 'rim_%s' % _side
            _TGT_STRUCT = 'pelvis_%s' % _side
            _RIM_STRUCT = 'rim_%s' % _side
            _COMP_STRUCT = '%s_%s' % (_organ2, _side)
            # _COMP_STRUCT = '%s_%s' % (_organ2, _side)
            _SRC_STRUCT = '%s_%s' % (_organ1, _side)
            _opo_pelvis = 'pelvis_%s' % _sides[i]
            # tag = '-vessels_label_lr'
            tag = ''
            print(_COMP_STRUCT)
            print(_SRC_STRUCT)

            if 'vein'  in _SRC_STRUCT:
                _SUB_COLORS = [
                    # (0.71, 0.62, 0.14),  # nerve
                    (0.8, 0.45, 0.25), # artery
                    (0.9, 0.9, 0.9),  # _opo_pelvis
                    (0.9, 0.9, 0.9)
                ]
            if 'nerve' in _SRC_STRUCT:
                _SUB_COLORS = [
                    (0.8, 0.45, 0.25),  # artery
                    (0.0, 0.6, 0.8),  # vein
                    # (0.8, 0.45, 0.25), # artery
                    # (0.71, 0.62, 0.14),  # nerve
                    (0.9, 0.9, 0.9),# _opo_pelvis
                    (0.9, 0.9, 0.9),
                ]
            elif 'artery' in _SRC_STRUCT:
                _SUB_COLORS = [
                    (0.0, 0.6, 0.8),  # vein
                    # (0.71, 0.62, 0.14),  # nerve
                    # (0.8, 0.45, 0.25), # artery
                    (0.9, 0.9, 0.9),# _opo_pelvis
                    (0.9, 0.9, 0.9)
                ]

            print(_cases)
            dists = dict()
            risk = dict()
            for _case in tqdm.tqdm(_cases):
                # _SRC = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _SRC_STRUCT))
                # _TGT = os.path.join(_POLY_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))

                # _SRC = os.path.join(_SEG_POLY_ROOT, '%s%s_%s_.vtk' % (_case,tag, _SRC_STRUCT))
                # vessel(vein/artery)_side
                _SRC = os.path.join(_GT_POLY_ROOT, '%s%s_%s.vtk' % (_case, tag, _SRC_STRUCT))
                # rim_side
                _TGT = os.path.join(_Pelvis_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))
                _RIM = os.path.join(_Rim_ROOT, '%s_%s.vtp' % (_case, _RIM_STRUCT))
                # 'X:/mazen/Segmentation/Data/HipMusclesDataset/Polygons/k1565_femur.vtk',
                _SUB_ACTORS = [os.path.join(_GT_POLY_ROOT, '%s_%s.vtk' % (_case, _COMP_STRUCT))]
                               # os.path.join(_Pelvis_ROOT, '%s_%s.vtk' % (_case, _opo_pelvis))]
                               # os.path.join(_Pelvis_ROOT, '%s_%s.vtk' % (_case, _TGT_STRUCT))]
                # os.path.join(_GT_POLY_ROOT, '%s_acetabulum.vtk' % (_case)),
                # os.path.join(_GT_Pelvis_ROOT, '%s_pelvis.vtk' % (_case))]
                for az in AZs:
                    min_dist = VisualizeSurfaceDistance(_SRC, _TGT, _RIM,
                                                            _ASIS_landmark, _IT_landmark,
                                                            case=_case,
                                                    sub_tgt_poly_path=_SUB_ACTORS,
                                                    sub_tgt_poly_colors=_SUB_COLORS,
                                                    sub_tgt_poly_alphas=_SUB_ALPHAS,
                                                    az=az,
                                                    tag=_side)
                dists[_case] = min_dist



            print(dists)
            df = pd.DataFrame(list(dists.items()), columns=['ID', _SRC_STRUCT]).set_index('ID')
            print(df)
            df.to_csv(os.path.join(_FIG_TGT, '%s.csv' % _SRC_STRUCT))




