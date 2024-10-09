import osmnx as ox
import pandas as pd

proxies = {
"http": "127.0.0.1:7890",
"https": "127.0.0.1:7890",
}
ox.settings.requests_kwargs['proxies'] = proxies

# 深圳市
# 经度均值: 22.558552301846003
# 纬度均值: 114.02637773797504
# 利用边界经纬度获取四边形区域的路网数据
G = ox.graph_from_address('22.508552301846003, 114.00637773797504',dist=5000,dist_type='network',network_type='drive')   # drive表示机动车道

# Step 3: 可视化该区域的路网
ox.plot_graph(ox.project_graph(G))

node_list = list(G.nodes(data=True))
print(len(node_list))