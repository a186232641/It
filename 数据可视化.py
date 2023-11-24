import numpy as np
import matplotlib.pyplot as plt

# 加载.npy文件
pred = np.load('results/test_iTransformer_ETTh1_MS_ft120_sl24_ll24_pl128_dm6_nh2_el2_dl1024_df5_fctimeF_ebTrue_dttest_projection_0_3/pred.npy')
true = np.load('results/test_iTransformer_ETTh1_MS_ft120_sl24_ll24_pl128_dm6_nh2_el2_dl1024_df5_fctimeF_ebTrue_dttest_projection_0_3/true.npy')
# 检查数据的形状来决定如何可视化
print("Data shape:", true.shape)

