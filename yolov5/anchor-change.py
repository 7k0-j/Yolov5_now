import utils.autoanchor as autoAC
# 对数据集重新计算 anchors
new_anchors = autoAC.kmean_anchors('./data/3k_data.yaml', 9, 1280, 4.0, 1000, True)
print(new_anchors)
