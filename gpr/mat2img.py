#!/usr/bin/env python
# coding: utf-8

# 将mat文件一一对应生成png文件
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 文件路径
mat_path, img_path = [], []
for i in os.listdir(os.path.join('gpr','A')):
    mat_path.append(os.path.join('gpr','A',i))
    img_path.append(os.path.join('gpr_img','A',i))
    mat_path.append(os.path.join('gpr','B',i))
    img_path.append(os.path.join('gpr_img','B',i))

# 保存图像
def save_img(mat_index, img_index):
    df = loadmat(mat_index)
    data = df['E_obs']
    
    fig = plt.gcf()
    fig.set_size_inches(12.0/3,8.0/3) #dpi = 300, output = 1200*800 pixels
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    plt.imshow(data[200:][:], vmin=-0.15, vmax=0.1, aspect='auto', cmap='jet')
    
    plt.savefig(img_index,format='png', transparent=True, dpi=300, pad_inches = 0)
    #plt.show()

def trans2img(mat_path, img_path):
    mat_name = os.listdir(mat_path)
    index = [x[:-4] for x in mat_name]
    img_name = [x+'.png' for x in index]
    # 一一对应的名称
    mat_list = [os.path.join(mat_path,x) for x in mat_name]
    img_list = [os.path.join(img_path,x) for x in img_name]
    for i in range(len(mat_list)):
        save_img(mat_list[i], img_list[i])
    
    mat_list = None
    img_list = None
    return

if __name__ == '__main__':
    for i in range(len(mat_path)):
        trans2img(mat_path[i], img_path[i])
        print('----------{} is transformed to {}---------'.format(mat_path[i], img_path[i]))



