import numpy as np
import osmnx as ox
import pandas as pd
import pickle

proxies = {
"http": "127.0.0.1:7890",
"https": "127.0.0.1:7890",
}
ox.settings.requests_kwargs['proxies'] = proxies

n_x, n_y = 16, 16  # 设置网格为16x16

def create_grid_bounds(edge, n_x, n_y):
    xmin, ymin, xmax, ymax = edge
    dx = (xmax - xmin) / n_x
    dy = (ymax - ymin) / n_y

    grid_bounds = []
    for i in range(n_y):
        for j in range(n_x):
            minx = xmin + j * dx
            miny = ymin + i * dy
            maxx = minx + dx
            maxy = miny + dy
            grid_bounds.append((minx, miny, maxx, maxy))
    return grid_bounds

# 计算网格的索引
def find_grid_index(x, y, grid_bounds):
    for idx, (minx, miny, maxx, maxy) in enumerate(grid_bounds):
        if minx <= x <= maxx and miny <= y <= maxy:
            return idx
    return None


# 计算网格内的节点数
def count_points_in_grid(points, edge, n_x, n_y):
    grid_bounds = create_grid_bounds(edge, n_x, n_y)
    grid_matrix = np.zeros((n_y, n_x), dtype=float)

    for point in points:
        x, y = point
        grid_index = find_grid_index(x, y, grid_bounds)
        if grid_index is not None:
            row = grid_index // n_x
            col = grid_index % n_x
            grid_matrix[row, col] += 1
        # else:
            # print(f"Point {point} is out of bounds.")

    return grid_matrix


# 计算需求矩阵的函数
def count_demand_in_grid(latitudes, longitudes, edge, n_x, n_y):
    grid_bounds = create_grid_bounds(edge, n_x, n_y)
    demand_matrix = np.zeros((n_y, n_x), dtype=float)

    for lat, lon in zip(latitudes, longitudes):
        grid_index = find_grid_index(lon, lat, grid_bounds)  # 经纬度顺序为(lon, lat)
        if grid_index is not None:
            row = grid_index // n_x
            col = grid_index % n_x
            demand_matrix[row, col] += 1
    return demand_matrix



if __name__ == '__main__':
    G = ox.graph_from_address('22.508552301846003, 114.00637773797504',dist=5000,dist_type='network',network_type='drive') # drive表示机动车道

    edge = ox.graph_to_gdfs(G=G, nodes=False).total_bounds
    print(edge)
    xmin, ymin, xmax, ymax = edge
    southwest = (xmin, ymin)
    northwest = (xmin, ymax)
    southeast = (xmax, ymin)
    northeast = (xmax, ymax)
    # coordinates for Hannover
    rec = [northwest, southwest, northeast, southeast]
    # 计算每个节点在哪个单元中，以及每个网格单元内部有多少节点(grid_density)
    # 获取路网的节点列表
    node_list = list(G.nodes(data=True))

    points = [(node[1]['x'], node[1]['y']) for node in node_list]
    print(len(points))
    grid_matrix = count_points_in_grid(points, edge, n_x, n_y)
    print("Grid matrix:")
    print(grid_matrix)

    # Check if total counts match
    total_count = np.sum(grid_matrix)
    print(f"Total points counted: {total_count}")
    print(f"Total points provided: {len(points)}")

    # 获取充电需求矩阵
    # 读取CSV文件
    data = pd.read_csv('./data/guang.csv')

    # CSV文件中有lat和lon列
    latitudes = data['lat'].values
    longitudes = data['lon'].values

    # 将lat和lon列组合成 (lat, lon) 形式的列表
    trajectory_points = list(zip(latitudes, longitudes))
    # 计算需求矩阵
    demand_matrix = count_points_in_grid(trajectory_points, edge, n_x, n_y)
    # 将需求 *0.05 为充电需求
    demand_matrix = np.round(demand_matrix * 0.05, 2)
    print("Grid Demand matrix:")
    print(demand_matrix)
    # 存储需求矩阵
    location = "ShenZhen"
    pickle.dump(demand_matrix, open("../../Graph/" + location + "/SZ_grid_demand" + ".pkl", "wb"))


