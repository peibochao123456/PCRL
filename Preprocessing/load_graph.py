import osmnx as ox
from shapely.geometry import Point, MultiPolygon, LineString, Polygon
from shapely.ops import split
import numpy as np
import pickle
import pandas as pd


proxies = {
"http": "127.0.0.1:7890",
"https": "127.0.0.1:7890",
}
ox.settings.requests_kwargs['proxies'] = proxies

"""
加载路网的结点数据
"""
n_x, n_y = 16, 16
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


def find_grid_index(x, y, grid_bounds):
    for idx, (minx, miny, maxx, maxy) in enumerate(grid_bounds):
        if minx <= x <= maxx and miny <= y <= maxy:
            return idx
    return None


def count_points_in_grid(node_list, edge, n_x, n_y):
    grid_bounds = create_grid_bounds(edge, n_x, n_y)
    grid_matrix = np.zeros((n_y, n_x), dtype=int)

    for node in node_list:
        x, y = node[1]['x'],node[1]['y']
        grid_index = find_grid_index(x, y, grid_bounds)
        if grid_index is not None:
            row = grid_index // n_x
            col = grid_index % n_x
            grid_matrix[row, col] += 1
            node[1]["row"] = row
            node[1]["column"] = col
        else:
            print(f"Point {node} is out of bounds.")

    return grid_matrix,node_list

if __name__ == '__main__':
    G = ox.graph_from_address('22.508552301846003, 114.00637773797504',dist=5000,dist_type='network',network_type='drive')  # drive表示机动车道

    edge = ox.graph_to_gdfs(G=G,nodes=False).total_bounds
    print(edge)
    # 计算每个节点在哪个单元中，以及每个网格单元内部有多少节点(grid_density)
    # 获取路网的节点列表
    node_list = list(G.nodes(data=True))
    # 获取密度 并为结点添加对应的行和列
    grid_density,node_list = count_points_in_grid(node_list, edge, n_x, n_y)
    print("grid_density:")
    print(grid_density)
    # Check if total counts match
    total_count = np.sum(grid_density)
    print(f"Total points counted: {total_count}")
    print(f"Total points provided: {len(node_list)}")

    print("Grid density is calculated.")

    # Save the graph files
    location = "ShenZhen"
    ox.save_graphml(G, filepath="../Graph/" + location + "/" + location + ".graphml")
    with open("../Graph/" + location + "/SZ_node_list" + ".txt", 'w') as file:
        file.write(str(node_list))

    # 密度矩阵
    pickle.dump(grid_density, open("../Graph/" + location + "/SZ_grid_density" + ".pkl", "wb"))

