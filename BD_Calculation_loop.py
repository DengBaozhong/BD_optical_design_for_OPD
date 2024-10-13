# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:03:01 2024

@author: maxim
"""

import numpy as np
import itertools
import BD_functions_multilayer_sample as fm
from multiprocessing import Pool
from tqdm import tqdm

def process_combination(combo):
    n, d1, d2 = combo
    tabepsmat = ['Glass', 'ITO', 'PEDOTPSS', 'PM66eC91_12', 'BCP', 'Ag', n, 'Air']
    elayer = np.array([150, 30, 120, 10, d1, d2], dtype=np.float64)
    
    tab = fm.load_txt_files(np.arange(300, 1000+10*0.5, 10),tabepsmat)
    # nmed = np.size(tabepsmat)
    result_absorption = fm.main(tab, 3, elayer, 300, 1000, 10, 0)
    
    return result_absorption, n, d1, d2

if __name__ == '__main__':
    range_n = np.arange(1, 3.1, 0.1)
    range_d1 = np.concatenate((np.arange(10, 40, 1), np.arange(40, 100, 5)), axis=0)
    range_d2 = np.arange(0, 200.1, 2)
    
    # Using itertools.product to generate combinations without creating a large meshgrid
    combinations = np.array(list(itertools.product(range_n, range_d1, range_d2)))
    
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

    # 一次性写入到 TXT 文件
    np.savetxt('results.txt', results_array.reshape(-1, results_array.shape[-1]))
