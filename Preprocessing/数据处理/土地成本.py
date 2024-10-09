import numpy as np
import pickle
import sys


np.random.seed(50)
estateprice = np.full((16, 16), 10)
print(estateprice)

location = "ShenZhen"
pickle.dump(estateprice, open("../../Graph/" + location + "/SZ_estateprice" + ".pkl", "wb"))

print('土地成本设计完毕')