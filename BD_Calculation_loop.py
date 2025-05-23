# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:03:01 2024

@author: maxim
"""

import numpy as np
import itertools
import BD_functions_multilayer as fm
from multiprocessing import Pool
from tqdm import tqdm

def process_combination(combo):
    n1, n2, d1, d2 = combo
    tabepsmat = ['Air',n1,n2,'Ag','Au','MoO3','PM6ec9','ZnO','Si']
    elayer = np.array([d1,d2,15,2.5,15,200,50], dtype=np.float64)
    
    BHJ = tabepsmat.index('PM6ec9')     #BHJ的位置
    
    lambmin = 400.
    lambmax = 850.
    dlamb = 10.
    
    tab = fm.load_txt_files(np.arange(lambmin, lambmax+dlamb*0.5, dlamb),tabepsmat)
    
    EQE = fm.main(tab, BHJ, elayer, lambmin, lambmax, dlamb, 0)
    
    # 筛选400-700nm范围内的数据
    mask = (EQE[:,0] >= 400) & (EQE[:,0] <= 700)
    EQE_400_700 = np.average(EQE[mask, 1])  # 计算筛选范围内第二列的平均值
    
    # 筛选800-850nm范围内的数据
    mask = (EQE[:,0] >= 800) & (EQE[:,0] <= 850)
    EQE_800_850 = np.average(EQE[mask, 1])
    
    return n1, n2, d1, d2, EQE_400_700, EQE_800_850

if __name__ == '__main__':
    range_n1 = np.arange(1, 3.1, 0.1)
    range_n2 = np.arange(1, 3.1, 0.1)
    range_d1 = np.arange(0, 200.1, 10)
    range_d2 = np.arange(0, 200.1, 10)
    
    # Using itertools.product to generate combinations without creating a large meshgrid
    combinations = np.array(list(itertools.product(range_n1, range_n2, range_d1, range_d2)))
    
    num_combinations = combinations.shape[0]
    print (f"There are {num_combinations:.0f} points.")
    
    results = []
    with Pool(processes=12) as pool:
        try:
            with tqdm(total=len(combinations)) as pbar:
                for combo in combinations:
                    result = pool.apply_async(process_combination, (combo,), 
                                              callback=lambda _: pbar.update(1))
                    results.append(result)
                # 确保所有任务完成
                pool.close()  # 不再接受新的任务
                pool.join()   # 等待所有工作完成

        except KeyboardInterrupt:
            print("计算中断，正在关闭进程...")
            pool.terminate()
            pool.join()

    results_array = np.array([res.get() for res in results])  # 获取所有结果

    # 写入到CSV文件（逗号分隔，保留浮点数精度）
    np.savetxt('results.csv', 
               results_array.reshape(-1, results_array.shape[-1]),
               delimiter=',',  # 明确指定逗号分隔
               fmt='%.6f',     # 控制浮点数格式（可选）
               header='n1, n2, d1, d2, EQE400~700, EQE800~850',  # 列名
               comments='')    # 避免header以#开头（可选）

