import glob
import os
import sys
sys.path.append('../')
sys.path.append('/public/home/acvm651ob1/cage-gen4/')
import pandas as pd
from pymatgen.core.structure import Structure
import numpy as np
from itertools import product
from scipy.spatial.distance import squareform, pdist
import networkx as nx
# from itertools import permutations
from sklearn.cluster import DBSCAN
from simple_path import simple_path
from sklearn.neighbors import KDTree as skKDTree
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree as scKDTree

"""ver3: 增加了通路的筛选条件"""
"""ver4: 增加了判断结构是否是完全笼子组成的判断逻辑"""
"""ver4.1: 逻辑修正, 但是不保证完全成功"""
"""ver5: make mesh换成kdtree逻辑; 增加笼子原子密度判定条件; 优化mesh团簇分类逻辑"""

def xyz_to_abc(vector, lattice):
    """lattice是POSCAR上面的lattice的转置, 是abc基矢在xyz空间的表示  (a,b,c)=M-1@(x,y,z)"""
    lattice_inv = np.linalg.inv(lattice)
    return lattice_inv @ vector


def abc_to_xyz(vector, lattice):
    """lattice是POSCAR上面的lattice的转置 (x,y,z)=M@(a,b,c)"""
    return lattice @ vector


def get_surface_vectors(vec):
    """vec是法向量, 实际上就是三维平面方程的系数(A,B,C), 返回的是过原点的平面矢量, xyz空间的"""
    hkl = [float(i) for i in vec]
    if hkl.count(0) == 3:
        return False
    if hkl.count(0) == 2:  # (1,0,0)(0,1,0)(0,0,1)的情况
        v1 = [1 if hkl[i] == 0 else 0 for i in range(len(hkl))]
        v2 = v1[:]
        v2[v1.index(1)] = -1
    elif hkl.count(0) == 1:  # (1,0,1)之类的情况
        id = np.where(np.array(hkl) != 0)[0]
        v1 = [1, 1, 1]
        v2 = [1, 1, 1]
        v1[id[0]] = -hkl[id[1]] / hkl[id[0]]
        v2[id[1]] = -hkl[id[0]] / hkl[id[1]]
        v2[np.where(np.array(hkl) == 0)[0][0]] = np.random.randint(2, 10)
    else:
        A, B, C = hkl[0], hkl[1], hkl[2]  # 任意非零的情况
        v1 = np.array([-B / A, 1, 0]).reshape(3, 1)
        v2 = np.array([0, 1, -B / C]).reshape(3, 1)  # xyz空间的
    return v1, v2


def half_ceil_floor(x, flag):
    if flag == 'ceil':
        xx = np.ceil(x)
        if xx - x >= 0.5:
            return xx - 2
        else:
            return xx - 2
    if flag == 'floor':
        xx = np.floor(x)
        if x - xx >= 0.5:
            return xx + 2
        else:
            return xx + 2


"""找到结构的bubbles"""


def checkHdensity(positions, point_mean, stru):
    """
    1. 扩胞3*3*3 保险起见
    2. 打网格, 网格范围是1/3-2/3, 保证所有的网格点周围都有原子
    3. mesh平移以point_mean为中心
    """
    lattice_matix = stru.lattice.matrix
    dif_a = 0.5/lattice_matix[0,0]  # 确定a轴方向0.5距离的分数值
    dif_c = 0.5/lattice_matix[2,2]  # 确定c轴方向0.5距离的分数值

    # xmin, ymin, zmin = np.min(positions, axis=0)
    # xmax, ymax, zmax = np.max(positions, axis=0)
    x_mesh = np.arange(1/6, 5/6, dif_a, dtype=np.float64)
    y_mesh = np.arange(1/6, 5/6, dif_a, dtype=np.float64)
    z_mesh = np.arange(1/6, 5/6, dif_c, dtype=np.float64)


    all_positions = stru.cart_coords
    bubble_points = []
    all_bubble_idx = []
    points = product(x_mesh, y_mesh, z_mesh)
    pp_frac = np.asarray(list(points)).reshape(-1, 3)
    pp = pp_frac @ lattice_matix  # 网络转换成直角坐标
    mesh_center = np.mean(pp, axis=0)
    shift_dist = mesh_center - point_mean
    pp = pp - shift_dist  # 把网络平移到中心放正
    """先以all_positions和mesh points两个整体判断, 非常快"""
    pp = kdtree(all_positions,pp,r=0.99)
    if len(pp) == 0:
        print('没有找到合适bubble')
        return bubble_points, all_bubble_idx, False
    """在对剩余的mesh点分类判断,检查是否有聚集的团簇,这样子提前一步判断"""
    core_coords, channel_flag = get_mean_core_coords(pp,all_positions)  # 这一步的core_coords也许可以替代pp, 测试一下看看
    if channel_flag:
        print('该结构存在通道, 跳过该结构')
        return bubble_points, all_bubble_idx, channel_flag
    core_coords = np.concatenate(core_coords).reshape(-1,3)
    """最后再一个一个bubble point的找"""

    # 泡泡应该满足最近的H距离大于1.4 使用KDtree方法快一点
    for p in core_coords:
        temp_bubble_idx, temp_bubble_coords = get_bubble_points(p, all_positions)  # 极限密度应该是3.3的立方体里面间隔0.8, 一共70个, 保守以100个为界限, 另外H-H距离<0.8的如果超过20个也不行
        if len(temp_bubble_idx) == 0 or len(temp_bubble_idx) > 100:
            continue
        dis_mat = cdist(temp_bubble_coords, temp_bubble_coords)
        np.fill_diagonal(dis_mat, 1)
        small_bond_number = np.sum(dis_mat <= 0.8)  # H-H<0.8的数量
        if small_bond_number > 20:
            continue
        if temp_bubble_idx:  # 泡泡点满足存在H距离在 1.4<2.3 之间, 并且H的数量大于3个
            bubble_points.append(p)
            all_bubble_idx.append(temp_bubble_idx)
    """使用kdtree+动态删除[动态删除会影响后面的cluster分类]快一倍"""
    # while True:
    #     if len(pp) == 0:
    #         break
    #     p = pp[0].reshape(1,3)
    #     pp = np.delete(pp,[0],axis=0)
    #     temp_bubble_idx, _ = get_bubble_points(p, all_positions)
    #     if temp_bubble_idx:  # 泡泡点满足存在H距离在 1.4<2.3 之间, 并且H的数量大于3个
    #         bubble_points.append(p)
    #         all_bubble_idx.append(temp_bubble_idx)
    #         print(len(bubble_points))  # 23报错
    #         if len(pp) == 0:
    #             break
    #         bubble_near_bubble_idx, bubble_near_bubble_coords = get_bubble_points(bubble_points,pp, r_max=1, h_flag=False)
    #         if bubble_near_bubble_idx:
    #             bubble_near_bubble_idx = list(set(bubble_near_bubble_idx))
    #             pp=np.delete(pp,bubble_near_bubble_idx,axis=0)  # 如果p点是一个bubble中心, 那么它bubble范围的其它p点就不需要进行判断了
    # debug_bubble_points = np.concatenate(bubble_points,axis=0)
    return bubble_points, all_bubble_idx, channel_flag  # 如果bubble_points为空, 就说明不是笼子


def get_bubble_points(bubble_point, coords, r_min=1, r_max=3.3, h_flag=True):
    """
    给一个坐标, 获得该坐标bubble内的点
    """
    if type(bubble_point) == list:
        bubble_point = np.concatenate(bubble_point)

    kd_tree_H_coord = skKDTree(coords)
    bubble_point = bubble_point.reshape(-1,3)
    n = kd_tree_H_coord.query_radius(bubble_point, r_max, return_distance=True, sort_results=True)  # kdtree方法找近邻原子
    # 多个bubble point的情况考虑
    idx = [n[0][i].tolist() for i in range(len(n[0]))]
    dist = [n[1][i].tolist() for i in range(len(n[1]))]
    idx = [i for subid in idx for i in subid]  # 近邻索引
    dist = [i for subid in dist for i in subid]  # 近邻距离
    if h_flag:  # 寻找H坐标的情况
        if len(idx) < 4 or min(dist) < r_min:  # 没有近邻; 最近的小于最小距离; 如果这个笼子表面非常的密集, 也不能要
            return [], []
    if len(idx) == 0:
        return [], []
    return idx, coords[idx]


def read_data(file):
    """读取结构文件, 获得lattice和h_coord"""
    """读取vasp结构"""
    a, b, c, lattice, h_coord = None, None, None, None, None
    if 'vasp' in file:
        with open(file) as f:
            lines = [i.strip() for i in f.readlines()]
            a = np.asarray([float(i) for i in lines[2].split()]).reshape(3, 1)
            b = np.asarray([float(i) for i in lines[3].split()]).reshape(3, 1)
            c = np.asarray([float(i) for i in lines[4].split()]).reshape(3, 1)
            lattice = np.concatenate([a, b, c], axis=1)
            h_coord = []
            for i in range(8, len(lines)):
                h_coord.append([float(j) for j in lines[i].split()[:3]])
            h_coord = np.asarray(h_coord).reshape(-1, 3)
    """读取xyz结构"""
    if 'xyz' in file:
        h_coord = np.loadtxt(file, skiprows=2, usecols=[1, 2, 3])
        a, b, c = 0, 0, 0
    return a, b, c, lattice, h_coord


def get_project_coord(stru):

    h_coord = stru.cart_coords
    """获取最中间的那一组三维空间坐标"""
    point_mean = np.mean(h_coord, axis=0)  # 确定中心点
    primi_atoms = int(len(h_coord) / 27)  # 原胞的原子数, 应该就是len(point2)/27
    """把坐标分成27组, 看哪一组的坐标离中心最近, 就以它的四边作为网格"""
    groups = []  # 用来放27组坐标的平均值 因为是3*3*3的扩胞
    for i in range(27):
        temp_idx = [j * 27 + i for j in range(primi_atoms)]
        temp_points = h_coord[temp_idx]
        temp_mean = np.mean(temp_points, axis=0)
        temp_dist = np.linalg.norm(temp_mean - point_mean)
        groups.append(temp_dist)
    center_group = np.argmin(groups)  # 通过判断不同胞的中心与中心点的距离, 选最近的那个
    primi_idx = [i * 27 + center_group for i in range(primi_atoms)]  # 包含中心点的一组胞
    center_points = h_coord[primi_idx]  # 以这组坐标的极值做网络
    center_first_coord = center_points[0]  # 这是原胞的平移量
    center_coord = np.mean(center_points, axis=0)
    bubble_points, bubble_sphere, channel_flag = checkHdensity(center_points, point_mean, stru)  # 画网格, 在网格中找到bubble点
    return bubble_points, bubble_sphere, center_first_coord, primi_idx, channel_flag


"""判断第4个点是否在平面"""


def check_gongxian(points):
    p1, p2, p3 = points[0], points[1], points[2]
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    ab = p2 - p1
    ac = p3 - p1
    if np.all(np.cross(ab, ac)) != 0:
        return True  # True表示不共线


def define_area(point1, point2, point3):
    """
    法向量    ：n={A,B,C}
    空间上某点：p={x0,y0,z0}
    点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=Ax+By+Cz-(Ax0+By0+Cz0)
    :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = np.asmatrix(point2 - point1)
    AC = np.asmatrix(point3 - point1)
    N = np.cross(AB, AC)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D


def point2area_distance(points, point4):
    """
    点到面的距离
    """
    p1, p2, p3 = points[0], points[1], points[2]
    Ax, By, Cz, D = define_area(p1, p2, p3)
    mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d


def check_edge(edge):
    curve_idx = list(set(edge))  # 这个环的坐标索引
    curve_coord = [h_coord[i] for i in curve_idx]  # 这个环的坐标
    for i in range(len(curve_coord) - 3):  # 这是判断共线的循环, 通常情况只需要运行一次
        if check_gongxian(curve_coord[i:i + 3]):
            for j in range(i + 3, len(curve_coord)):  # 这是判断其他点到平面的距离, 如果出现一次不满足, 就要退出这个edge的搜索
                point_dist = point2area_distance(curve_coord[i:i + 3], curve_coord[j])
                if point_dist > 0.8:  # 远离平面的距离
                    return False
            return True

def kdtree(coord1,coord2,r=1.):
    """以coord1为框架, 找到coord2中所有与coord1有距离差的坐标"""
    kd1 = scKDTree(coord1)
    kd2 = scKDTree(coord2)
    n = kd1.query_ball_tree(kd2,r=r)
    nouse_idx = [i for subset in n for i in subset]  # 这是距离H太近的网格, 直接删掉不考虑
    nouse_idx = list(set(nouse_idx))
    new_coords = np.delete(coord2,nouse_idx,axis=0)
    return new_coords

def max_distance_between_coords(coords):
    """一个数组的最远距离"""
    dis_mat = cdist(coords,coords)
    np.fill_diagonal(dis_mat,0)
    max_dist=np.max(dis_mat)
    return max_dist

def get_mean_core_coords(bubble_points, all_positions=None, iter_flag=0):
    """
    把在一起的bubble点分类取中心值. 减并
    """
    channel_flag = False
    if len(bubble_points) == 0:  # 这是在iter的时候, 会出现平移后找不到符合条件的坐标
        return [], channel_flag
    cluster = DBSCAN(eps=0.51, min_samples=1).fit(bubble_points)
    labels = cluster.labels_
    cores = list(set(labels))
    core_coords = []
    for i in range(len(cores)):
        temp_core_coord = [bubble_points[j] for j in range(len(labels)) if labels[j] == cores[i]]  # 这里面记录了每一类的坐标数量
        debug_temp_core_cood = np.concatenate(temp_core_coord).reshape(-1,3)
        max_dist = max_distance_between_coords(temp_core_coord)

        if max_dist > 5 and iter_flag == 0:  # 如果连通的点群最远距离超过笼子的长度, 就说明它有问题了, 前提要保证mesh撒的点都在超胞内部
            # 需要在判断一下是否存在割点, 还是再加一组平移的mesh判断呢?
            temp_all_coords = all_positions + np.array([0, 0, 0.3]).reshape(1, 3)
            new_temp_coord_coord = kdtree(temp_all_coords, temp_core_coord)
            temp_core_coords, channel_flag = get_mean_core_coords(new_temp_coord_coord, all_positions, iter_flag=1)
            if channel_flag:
                break
            if len(temp_core_coords) == 0:
                continue
            else:
                core_coords.extend(temp_core_coords)
                continue
        if max_dist > 5 and iter_flag == 1:
            print(f'iter_flag:1 发现bubble_points团簇长度超过5 {max_dist}, 再检查一下')
            temp_all_coords = all_positions + np.array([0.3, 0, 0]).reshape(1, 3)
            new_temp_coord_coord = kdtree(temp_all_coords, temp_core_coord)
            temp_core_coords, channel_flag = get_mean_core_coords(new_temp_coord_coord, temp_all_coords, iter_flag=2)
            if len(temp_core_coords) == 0:
                return [], channel_flag  # iter 1的遍历中, 只要发现一个大团簇, 就返回空
            else:
                core_coords.extend(temp_core_coords)
                continue
        if max_dist > 5 and iter_flag == 2:
            print(f'iter_flag:2 发现bubble_points团簇长度超过5 {max_dist}, 忽略该点, be caution')
            channel_flag = True
            return [], channel_flag
        temp_core_coord = np.mean(np.asarray(temp_core_coord).reshape(-1, 3), axis=0)  # 如果是一大块连通的, 直接取平均是不合适的
        core_coords.append(temp_core_coord)
    return core_coords, channel_flag  # core_coords是坐标, labels是分类的组


def mk_xyz(coords, test_coords, nameflag=None, index=False, multi=False):
    """
    用来产生xyz, debug用
    """
    # core_coords, labels = get_mean_core_coords(test_coords)
    elements = {0: 'B', 1: 'C', 2: 'N', 3: 'O', 4: 'F', 5: 'P', 6: 'S', 7: 'Cl'}
    """把coord和index区分转化, 这样子input无论是index还是coord都可以"""
    if index:
        test_coords = coords[test_coords]

    with open(f'{nameflag}.xyz', 'w') as f:
        if multi:
            total_atoms_num = sum([len(i) for i in test_coords]) + len(coords)
            print(f'{total_atoms_num}', file=f)
        else:
            print(f'{len(coords) + len(test_coords)}', file=f)
        print('debug', file=f)
        for i in coords:
            print(f'H  {i[0]}  {i[1]}  {i[2]}', file=f)
        if multi:
            for j in range(len(test_coords)):
                temp_coord = test_coords[j]
                for k in temp_coord:
                    print(f'{elements[j]}  {k[0]}  {k[1]}  {k[2]}', file=f)
        else:
            for j in range(len(test_coords)):
                print(f'C  {test_coords[j][0]}  {test_coords[j][1]}  {test_coords[j][2]}', file=f)
            # print(f'{elements[labels[j]]}  {test_coords[j][0]}  {test_coords[j][1]}  {test_coords[j][2]}', file=f)
        # for k in range(len(core_coords)):
        #     print(f'Li  {core_coords[k][0]}  {core_coords[k][1]}  {core_coords[k][2]}', file=f)


"""判断点之间的夹角"""


def angle(v1, v2):
    v1 = v1.reshape(3, )
    v2 = v2.reshape(3, )
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)
    return angle


def get_angle_matrix(core_coord, white_list):
    """
    获得所有点相对中心点的夹角角度, 以及夹角小于16的原子对
    """
    coords = h_coord[white_list]
    core_coord = core_coord.reshape(1, 3)
    coords = coords.reshape(-1, 3)
    angle_matrix = np.zeros(shape=(coords.shape[0], coords.shape[0]))
    for i in range(angle_matrix.shape[0]):
        for j in range(angle_matrix.shape[1]):
            if i == j:
                angle_matrix[i][j] = 100  # 两个点的夹角
                continue
            pa = coords[i] - core_coord
            pb = coords[j] - core_coord
            angle_ab = angle(pa, pb)
            angle_matrix[i][j] = angle_ab  # 两个点的夹角
    near_pairs = np.where(angle_matrix < 16)
    near_pairs = [sorted((near_pairs[0][i], near_pairs[1][i])) for i in range(len(near_pairs[0]))]
    near_pairs = [eval(i) for i in list(set([str(i) for i in near_pairs]))]  # 这里是所有夹角小于16度的端点对
    near_pairs = [[white_list[i[0]], white_list[i[1]]] for i in near_pairs]  # 把角度矩阵的索引变成对应h_coord的
    return angle_matrix, near_pairs


def get_core_dist_matrix(core_coord, coords, near_pairs):
    """
    获得中心点到其他点的距离, 应该只需要考虑angle里面小于16度的点就可以
    """
    black_list = []  # 这里放距离大的那一个索引
    for pair in near_pairs:
        coord_a = coords[pair[0]]
        coord_b = coords[pair[1]]
        dist_a = np.linalg.norm(coord_a - core_coord)
        dist_b = np.linalg.norm(coord_b - core_coord)
        if dist_a < dist_b:
            black_list.append(pair[1])
        else:
            black_list.append(pair[0])
    return list(set(black_list))


"""进一步筛选端点都在这里了"""


def refine_sphere_points(core_points, h_coord, core_coord):
    """
    进一步筛检circles,需要知道core point的坐标, 然后把所有的环的点距离求出来, 获得一个白/黑名单, 包含黑名单的环都不要
    """

    def _connect_three(_white_list):
        """
        计算list里面连接数, 剔除<3的点; bond距离是1.4
        """
        temp_coord = h_coord[_white_list]
        temp_distmat = get_distmat(temp_coord)  # 连接矩阵
        temp_connect_num = np.sum(temp_distmat, axis=1)
        temp_black_list = [_white_list[i] for i in range(len(temp_connect_num)) if temp_connect_num[i] < 3]

        return temp_black_list  # 这是连接数小于3个的端点list

    def _check_fengbi(_black_list):
        cage_flag = True
        while True:
            white_list = [i for i in all_circle_points if i not in _black_list]
            if len(white_list) == 0:
                # print('没有找到封闭端点')
                cage_flag = False
                break
            temp_black_list = _connect_three(white_list)
            if len(temp_black_list) == 0:
                # print('找到封闭端点')
                break
            else:
                _black_list.extend(temp_black_list)
        if len(all_circle_points) - len(_black_list) < 13:  # 要确保笼子的端点数大于13个
            cage_flag = False

        return _black_list, cage_flag

    def _less_than_eight(_white_list):
        bl = []
        edge_num = 7
        if len(_white_list) < 25:  # 在vertex总数小于25时, 允许八边存在
            edge_num = 8
        new_coords = h_coord[_white_list]
        dist_mat = get_distmat(new_coords)
        graph = nx.Graph(dist_mat)
        for i in range(dist_mat.shape[0]):
            path = list(simple_path.all_simple_paths(graph, source=i, target=i, cutoff=8))
            refine_path = [sorted(list(set(i))) for i in path if len(i) > 4]
            refine_path = [eval(j) for j in list(set([str(i) for i in refine_path]))]
            refine_path = sorted(refine_path, key=len)
            """剔除重复的大路径"""
            bks = []
            for a in refine_path:
                for b in refine_path:
                    same_len = len([ii for ii in a if ii in b])
                    if len(a) < len(b) and (same_len == len(a) or same_len > 3) and b not in bks:
                        bks.append(b)
            for bk in bks:
                refine_path.remove(bk)  # 有可能在之前的i端点就把b给删掉了

            path_len = [len(i) for i in refine_path]
            if len(path_len) < 3:
                bl.append(_white_list[i])
            else:
                path_len = sorted(path_len)[:3]
                for j in path_len:
                    if j > edge_num:
                        bl.append(_white_list[i])
        return bl

    def _get_long_short_ratio(_white_list):
        coords = h_coord[_white_list]
        bond_dist = []  # 用来放所有点到中心的距离
        refresh_core_coord = np.mean(coords, axis=0)
        for coord in coords:
            d = np.linalg.norm(coord - refresh_core_coord)
            bond_dist.append(d)
        shortest = np.min(bond_dist)
        longest = np.max(bond_dist)
        ratio = longest / shortest
        if ratio > 1.72:  # 暂定
            return False, ratio, refresh_core_coord
        else:
            return True, ratio, refresh_core_coord

    all_circle_points = core_points
    # all_circle_points_coords = h_coord[all_circle_points]

    """第一个筛选, 在大球里面, 这一步已经在get_bubble_points里面实现了"""
    black_list = []

    """第二个筛选, 所有点有至少三个最近邻也在大球内"""
    print('\n+++封闭性筛选',end=' ')
    black_list, cage_flag = _check_fengbi(black_list)
    white_list = [i for i in all_circle_points if i not in black_list]
    if not cage_flag:
        print('封闭性未通过')
        return False, None
    else:
        print(f'封闭端点, 共{len(white_list)}个')

    """第三个筛选, 点的夹角要大于16度"""
    print('\n+++原子夹角筛选',end=' ')
    angle_matrix, near_pairs = get_angle_matrix(core_coord, white_list)
    bl = get_core_dist_matrix(core_coord, h_coord, near_pairs)
    if len(bl) == 0:
        print('没有发现重叠原子')
    else:
        print('去除重叠原子')
    black_list = black_list + bl
    print('\n重新进行封闭性检查',end=' ')
    black_list, cage_flag = _check_fengbi(black_list)
    if not cage_flag:
        print('封闭性未通过')
        return False, None
    else:
        print('通过')
    white_list = [i for i in all_circle_points if i not in black_list]
    debug_white_coord = h_coord[white_list]
    """第四个筛选, 每个点最小的三个连接环长度小于8"""
    print('\n++++多边形环筛选',end=' ')
    bl = _less_than_eight(white_list)
    if len(bl) == 0:
        print('没有发现大于七边环')
    else:
        print('去除大于七边环原子')
        black_list = black_list + bl
    print('\n重新进行封闭性检查',end=' ')
    black_list, cage_flag = _check_fengbi(black_list)
    if not cage_flag:
        print('封闭性未通过')
        return False, None
    else:
        print('通过')
    white_list = [i for i in all_circle_points if i not in black_list]

    """第五个筛选, 距离中心的最近与最远距离比值限定"""
    print('\n+++++笼子最近距离与最远距离比值检查',end=' ')
    cage_flag, ratio, refresh_core_coord = _get_long_short_ratio(white_list)  # 这里应该先重置一下中心点的坐标
    if not cage_flag:
        print(f'第五步笼子距离比值未通过 {ratio}')
        return False, None
    else:
        print('通过')

    """第六个筛选, 检查通路. 所有的点要连通在一起"""
    coords_dist = squareform(pdist(h_coord[white_list]))
    coords_dist = np.where(coords_dist == 0, True, np.where((coords_dist >= 0.5) & (coords_dist <= 1.4), True, False))
    print('\n+++++最后一步, cage unit连通性检查',end=' ')
    msg = find_tonglu(coords_dist)  # 正常的笼子是所有连接在0.8-1.4之间, 我把标准放宽一些
    if msg is False:
        print('连通性未通过')
        return False, None
    else:
        print('通过')

    return white_list, refresh_core_coord


def refine_circles(edges, white_list):
    refine_edge = []
    for edge in edges:
        flag = True
        for i in edge:
            if i not in white_list:
                flag = False
                break
        if flag:
            refine_edge.append(edge)
    return refine_edge


def get_distmat(coords):
    """
    给定坐标list, 返回连接矩阵
    """
    dist_mat = squareform(pdist(coords))
    dist_mat = np.where(dist_mat == 0, 99, dist_mat)
    dist_mat = np.where(dist_mat < 1.4, 1, 0)  # 邻接矩阵
    return dist_mat


def read_xyz(file):
    """简易的读取xyz中H坐标"""
    data = np.loadtxt(file, skiprows=2, usecols=[1, 2, 3], comments='C')
    return data


"""判断bubble是否一样"""


def check_duplicate(bubble_types, white_list):
    """
    bubble_types是装有不一样的笼子的坐标列表; white_list是要判断的笼子的坐标索引. 通过检查端点数量和距离矩阵的特征值来判断是否一样
    """
    for bubble in bubble_types:
        if len(bubble) == len(white_list):
            bubble_distmat = squareform(pdist(bubble))
            cage_distmat = squareform(pdist(h_coord[white_list]))
            db = np.linalg.det(bubble_distmat)
            dc = np.linalg.det(cage_distmat)
            if abs(db - dc) < 0.01:  # 判断行列式不行, 有误差
                return True
    return False  # False表示不是重复的笼子


"""增加原胞结构读取接口, 替换read_data函数"""


def get_primit_stru(file):
    """
    读取结构, 然后记录第一个原子的坐标, 作为转换的锚点, 记录晶胞8个顶点的坐标, 顺便扩胞
    """
    stru = Structure.from_file(file)  # cif
    eles = [i for i in stru.species if i.name != 'H']
    try:
        stru.remove_species(eles)
    except:
        pass
    # stru.merge_sites(tol=0.51,mode='average')
    oa, ob, oc = stru.lattice.matrix[0], stru.lattice.matrix[1], stru.lattice.matrix[2]  # 这是三个点的坐标
    ab, ac, bc, abc = oa + ob, oa + oc, ob + oc, oa + ob + oc  # 这是八个顶点的坐标
    first_atom_coord = stru.cart_coords[0]
    stru.make_supercell([3, 3, 3])  # 这里扩三倍胞, 只需要他们的坐标就可以了
    coords_three = stru.cart_coords
    return coords_three, first_atom_coord, [oa, ob, oc, ab, ac, bc, abc], stru


"""判断bubble坐标是否在primit cell内部, 并转换坐标"""


def check_in_cell(bubble_coord, center_first_coord, first_atom_coord, vertexs, center_coord):
    """
    获得第一个原子的坐标平移量, 然后把vertex平移变换, 判断core_coord是否在平移变换后的立方体里面[通过判断它们与中心点的夹角]
    """
    center_coord = center_coord.reshape(3, )
    shift = center_first_coord - first_atom_coord  # 单胞的平移量
    bubble_coord_in_primit = bubble_coord - shift  # 虽然平移了, 但是不在第一个周期里面
    # 先判断core_coord和center_coord是不是同一个点
    if np.linalg.norm(bubble_coord - center_coord) < 0.1:
        return True, bubble_coord_in_primit

    shifted_vertex = [np.asarray(i + shift).reshape(3, ) for i in vertexs]  # 7个点
    shifted_vertex.append(np.asarray(shift).reshape(3, ))  # shift本身是(0,0,0)的平移坐标
    cc = center_coord - bubble_coord
    for i in shifted_vertex:
        ci = i - bubble_coord
        ang = angle(cc, ci)
        if ang > 90:
            return True, bubble_coord_in_primit
    return False, bubble_coord_in_primit


def mk_primit_poscar(primit_bubble_coord, primit_bubble_eles, file, nameflag):
    """
    重新读取file, 然后把bubble的坐标加进去, 元素按照primit_bubble_ele, primit_bubble_eles_type分类, 长度最小的放Li, 最大的放Na, 中间的放Ca; 5个笼子的中间都放Ca
    要剔除1埃以内的重叠坐标; 小笼子都用Li吧, 不必小于20
    """

    def _read_stru(file, bubble_coords, eles, read_cart=True, out_frac=True, write=False):
        stru = Structure.from_file(file)
        for i in range(len(bubble_coords)):
            temp_coord = bubble_coords[i]
            temp_ele = eles[i]
            stru.append(species=temp_ele, coords_are_cartesian=read_cart, coords=temp_coord)
        if out_frac:
            coords = stru.frac_coords[-len(bubble_coords):, :]
        else:
            coords = stru.cart_coords[-len(bubble_coords):, :]
        if write:
            # stru.merge_sites(tol=0.5, mode='average')  # 这是generate产生的结构专用, 一般也不需要
            print(f'++++++{file} 找到笼型单元 输出结构++++++')
            stru.to(filename=nameflag, fmt='poscar')
        return coords

    min_vertex = min(primit_bubble_eles)
    max_vertex = max(primit_bubble_eles)
    eles = ['Li' if i == min_vertex else 'Na' if i == max_vertex else 'Ca' for i in primit_bubble_eles]

    # step1 读取stru, 为了获得primit_bubble_coord的分数坐标, 放在单胞内
    primit_bubble_frac_coords = _read_stru(file, eles=eles, bubble_coords=primit_bubble_coord, read_cart=True, out_frac=True,
                                           write=False)
    while True:
        if np.all(primit_bubble_frac_coords>=0) and np.all(primit_bubble_frac_coords<=1):
            break
        primit_bubble_frac_coords = np.where(primit_bubble_frac_coords < 0, primit_bubble_frac_coords + 1,
                                             np.where(primit_bubble_frac_coords > 1, primit_bubble_frac_coords - 1,
                                                      primit_bubble_frac_coords))
    # step2 把分数坐标换成直角坐标
    primit_bubble_cart_coords = _read_stru(file, eles=eles, bubble_coords=primit_bubble_frac_coords, read_cart=False,
                                           out_frac=False, write=False)
    # step3 去重距离太近的, 小于1应该就不行了
    good_bubble=[primit_bubble_cart_coords[0]]
    good_eles = [eles[0]]
    for i in range(1,len(primit_bubble_cart_coords)):
        for j in good_bubble:
            if np.linalg.norm(primit_bubble_cart_coords[i]-j)<1:
                break
        else:
            good_bubble.append(primit_bubble_cart_coords[i])
            good_eles.append(eles[i])
    nameflag=nameflag+f'{len(good_eles)}.vasp'
    if len(good_eles) == 1:
        print(f'笼子去重 {len(eles)} -> {len(good_eles)} ')
        return good_eles,good_bubble
    _read_stru(file,bubble_coords=good_bubble,eles=good_eles,read_cart=True,write=True)
    return good_eles,good_bubble

"""最初筛"""


def check_H_network(file):
    """首先应该保证H-H的最近邻距离在0.7-1.25之间; 有的结构原子重合0.6埃以内, 要不要考虑merge, 可以先把这些结构单独提出来看一下"""
    stru = Structure.from_file(file)
    eles = [i for i in stru.species if i.name != 'H']
    try:
        stru.remove_species(eles)
    except:
        pass
    # H-H的距离应该都保持在0.7-1.25之间
    stru.make_supercell([2,2,2])
    H_dist_mat = stru.distance_matrix
    if np.any((H_dist_mat<0.7) & (H_dist_mat>0.2)):
        print('存在H-H距离小于0.7的点, be caution')
    H_dist_mat = np.where(H_dist_mat == 0, True, np.where((H_dist_mat >= 0.7) & (H_dist_mat <= 1.4), True, False))
    msg = find_tonglu(H_dist_mat)
    return msg


def find_tonglu(mat):
    """
    floyd方法判断通路
    nodes: list 用来装通路的节点数, 里面是索引, 长度是节点数
    :param mat: H的距离矩阵, 0.9-1.5的为True
    :return:
    """
    nodes = {i: [i] for i in range(mat.shape[1])}  # 总
    for i in range(mat.shape[1]):
        if mat[0][i] == True:
            nodes[i].append(0)
    temp_idx = []
    while True:
        idx = [i for i in range(mat.shape[0]) if mat[0][i] == True]
        not_idx = [i for i in range(mat.shape[0]) if i not in idx]
        idx_copy = [i for i in range(mat.shape[0]) if i not in not_idx]  # 备份不随着temp_idx更改
        if len(temp_idx) != 0:
            idx = temp_idx
            temp_idx = []
        for i in idx:
            if i == 0: continue
            for j in not_idx:  # 未找到通路的原子
                if mat[i][j] == True and mat[0][j] != True:
                    mat[0][j] = True
                    # temp_nodes[j].append(i)
                    nodes[j].extend(nodes[i])
                    temp_idx.append(j)  # 新找到的通路原子
        if len(temp_idx) == 0:  # 如果没找到通路, 说明已经找完了, 或者没有了
            break
    if np.any(mat[0] == False):
        msg = '无法构成完整通路'
        return False
    else:
        msg = 'H网络可以构成通路'
        return True


"""第一步: 读取3*3*3的结构, 寻找泡泡的坐标, 每一个泡泡进行立方体的寻找, 如果没有找到立方体则循环下一个泡泡, 直到结束"""
# log_flag=0

current_dir = os.path.dirname(os.path.abspath(__file__))
dir_name = os.path.basename(current_dir)
log_flag = dir_name
# with open(f'log-{log_flag}','r') as log_file:
#     lines=[i.strip() for i in log_file.readlines()]


good_stru = []
lines=[1]
for line in range(len(lines)):
    # file = lines[line]
    # file=f'../../{file}'
    # file='tt_spg_116-wp_2d-0-0-bond-network.vasp'
    # file='bt_spg_89-wp_4l-0-0-normal-network.vasp'
    # file='tb_spg_142-wp_8a-0-2-normal-network.vasp'
    # file='tt_spg_192-wp_2b-0-2-bond-network.vasp'
    file='bt_spg_101-wp_4c-0-2-center-network.vasp'
    # file='demo.vasp'
    fi = file.split('/')[-1].split('.')[0]
    print(f'======================================  {line}/{len(lines)}')
    try:
        msg = check_H_network(file)
        if not msg:
            print(f'{file} 未组成H网络')
            continue
        else:
            print(f'{file} 可以组成H网络')
    except:
        print('检出H连通性错误, 跳过')
        pass
    h_coord, first_atom_coord, vertexs, stru = get_primit_stru(file)  # h_coord是超胞的直角坐标; first_atom_coord是原胞的第一个点坐标, vertex是单胞8个顶点的坐标
    # a, b, c, lattice, h_coord = read_data(file)  # 读取已经3*3*3的supercell
    # h_coord = read_xyz(file)  # 读取xyz, 自动忽略非H原子坐标
    # dist_mat = get_distmat(h_coord)  # 只限制了 r<1.4
    # graph = nx.Graph(dist_mat)  # 图已经建立起来了, 这个好像没用了
    print('making mesh...')
    bubble_points, bubble_spheres, center_first_coord, center_coord_index, channel_flag = get_project_coord(stru)
    if channel_flag:
        cage_flag='channel'
    elif len(bubble_points) == 0:
        print('没有找到bubble')
        continue
    i = 0
    all_cages = []
    all_cages_in_center = []
    all_cages_center = []  # 判断是cage的中心点
    bubble_types = []  # 装不一样的bubble的坐标list
    primit_bubble_coord = []  # 用来装转换为primitcell的bubble坐标
    primit_bubble_eles = []  # 判断bubble所在的笼子大小, 来放Li或者Ca元素
    bubble_type_flag = 0  # 为了分开笼子vertex相同, 但是种类不同的情况
    primit_bubble_eles_type = []
    bubble_coord_in_primit = None  # 为了好看
    cage_in_center_index = []  # 用来在最后判断中心的点是否出现超过2次
    if not channel_flag:
        """其实bubble_points就已经可以初步把通道型筛出了, 现在看每一个bubble的原子"""
        core_coords_debug = np.concatenate(bubble_points).reshape(-1, 3)
        print(f'{file} 找到bubble共{len(core_coords_debug)}个, 第一个bubble的原子数量为{len(bubble_spheres[0])}')
        for i in range(len(bubble_points)):
            core_coord = bubble_points[i]
            core_bubble_sphere_points = bubble_spheres[i]
            print(f'正在寻找 core_coord {i+1}/{len(bubble_points)}  {core_coord} 初始端点数: {len(bubble_spheres[i])}', end=' ')
            debug_core_bubble_sphere_points = h_coord[core_bubble_sphere_points]
            """把core_bubble_spheres里面的点进一步筛选"""
            white_list, refresh_core_coord = refine_sphere_points(core_bubble_sphere_points, h_coord, core_coord)
            if not white_list:  # 如果没有找到封闭端点, 就该跳过这个core point了
                continue

            debug_coord=h_coord[white_list]
            all_cages.append(debug_coord)  # 所有的笼子结构坐标
            all_cages_center.append(core_coord.reshape(1,3))
            all_cages_center_coords = np.concatenate(all_cages_center,axis=0)
            """把所有的cage unit都收集起来, 然后找出其中有中心区域坐标的笼子, 最后整合到一起, 如果中心区域的点在里面出现超过2次, 则说明这个点属于至少两个笼子[正常应该是三个]"""
            cage_in_center_index.extend(white_list)


            # """这个函数不行, 以后再改"""
            # duplic_cage = check_duplicate(bubble_types, white_list)
            # if duplic_cage:
            #     print('笼子已经重复')
            #     primit_bubble_eles_type.append(bubble_type_flag)  # 不同种类的笼子按照1,2编号分类
            #     continue  # 如果是重复的则跳过这个结构
        """退出循环, 最后判断中心区域的点是否出现在两个以上cage中"""
        if len(all_cages) == 0:
            print('没有找到笼子')
            continue
        cage_flag='yes'
        debug_center_coord = h_coord[center_coord_index]
        all_cages_coords = np.concatenate(all_cages,axis=0)
        debug_all_cage_centers = np.concatenate(all_cages_center)
        tri_count, bi_count = 0, 0
        for center_coord_id in center_coord_index:
            if cage_in_center_index.count(center_coord_id) > 2:
                tri_count += 1
            else:
                bi_count += 1
        ratio = tri_count/len(center_coord_index)
        if ratio < 0.5:  # 缓兵之计, 假设有超过一半有三个笼子组成
            cage_flag='no'
            print(f'该结构可能不是完全由笼子组成 ratio_3: {ratio}')

    print(f'结果已保存 {cage_flag}-{file}')
    good_stru.append(f'{cage_flag}-{file}')  # 是笼子结构就添加
    """自动在笼子中心放金属元素得到poscar, 还需要调试"""
    # good_eles, good_bubble = mk_primit_poscar(primit_bubble_coord=primit_bubble_coord, primit_bubble_eles=primit_bubble_eles, file=file,
    #                  nameflag=rf'primit-{cage_spg}-{cage_name}-{cage_type}-{fi}-with-')
    # if len(good_eles) == 1:
    #     continue
    """对于一个cage unit, 外面多一个尖尖的情况, 要怎么做才能剔除那个尖尖的原子; 待更新 """
    """输出一个超胞xyz结构, 直观显示找到的笼子位置, 还需要调试"""
    #mk_xyz(h_coord, bubble_types, nameflag=f'multi-{fi}-with-{len(bubble_types)}', index=False,
    #       multi=True)  # 根据不同的bubble, 换不同的元素
    print(f'{file} 搜索结束')
print('程序结束')
with open(rf'{log_flag}-cage-log', 'w') as f:
    for i in good_stru:
        print(i, file=f)
    print('finished')