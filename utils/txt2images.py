# -*- coding: utf-8 -*-
import shutil
 
def objFileName():
    #local_file_name_list = "/data/private_data/jf_private/CCPD2020/CCPD2019/splits/test.txt"
    local_file_name_list = "/data/private_data/jf_private/CCPD2020/CCPD2020/ccpd_green/train.txt"
    obj_name_list = []
    for i in open(local_file_name_list, 'r'):
        obj_name_list.append(i.replace('\n', ''))
    return obj_name_list
 
def copy_img():
    local_img_name = "/data/private_data/jf_private/CCPD2020/CCPD2019/ccpd_base"
    # 指定要复制的图片路径
    path = "/data/private_data/jf_private/CCPD2020/CCPD2019/val"
    # 指定存放图片的目录
    for i in objFileName():
        new_obj_name = i.split('/')[1]
        shutil.copy(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)

def copy_img2():
    #local_img_name = "/data/private_data/jf_private/CCPD2020/CCPD2019"
    local_img_name = "/data/private_data/jf_private/CCPD2020/CCPD2020/ccpd_green/train"
    # 指定要复制的图片路径
    path = "/data/private_data/jf_private/CCPD2020/CCPD2019/train"
    # 指定存放图片的目录
    for i in objFileName():
        new_obj_name = i.split('/')[1]
        #print(new_obj_name)
        #shutil.copy(local_img_name + '/' + i, path + '/' + new_obj_name)
        shutil.copy(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)
 
if __name__ == '__main__':
    #train,val-1
    #copy_img()
    #test-2
    copy_img2()
