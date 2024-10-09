import pandas as pd

# 读取CSV文件
df = pd.read_csv('./data/guang.csv',nrows=1000000)

# 获取lat和lon的最大值及最小值
lat_min = df['lat'].min()
lat_max = df['lat'].max()
lon_min = df['lon'].min()
lon_max = df['lon'].max()

# 打印经纬度的最小值和最大值
print(f"纬度最小值: {lat_min}, 纬度最大值: {lat_max}")
print(f"经度最小值: {lon_min}, 经度最大值: {lon_max}")


# 计算经度和纬度列的均值
lon_mean = df['lon'].mean()
lat_mean = df['lat'].mean()

# 打印经纬度的均值
print(f"经度均值: {lon_mean}")
print(f"纬度均值: {lat_mean}")

# 构造四边形（按照顺时针或逆时针顺序）
rectangle = [
    (lat_min, lon_min),  # 左下角
    (lat_min, lon_max),  # 右下角
    (lat_max, lon_max),  # 右上角
    (lat_max, lon_min)   # 左上角
]
bbox = (lat_max, lat_min, lon_max, lon_min)
print(bbox)
# 打印四边形
print("四边形的四个顶点:", rectangle)
