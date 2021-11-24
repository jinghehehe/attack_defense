import cv2
import os, sys
path ="/data/private_data/jf_private/CCPD2020/CCPD2020/ccpd_green/train"

path_='/data/private_data/jf_private/CCPD2020/CCPD2020/ccpd_green/'
#lable=0
#item=0
txt = open(path_+'train.txt', 'w+')
for img_name in os.listdir(path):
    print(img_name)
    # 读取图片的完整名字
    # image = cv2.imread(path + "/" + img_name)
    # list_image=[]
    # # 以 - 为分隔符，将图片名切分，其中iname[4]为车牌字符，iname[2]为车牌坐标
    # iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    # tempName = iname[4].split("_")
    # # name = provinces[int(tempName[0])] + alphabets[int(tempName[1])] + ads[int(tempName[2])] \
    # #        + ads[int(tempName[3])] + ads[int(tempName[4])] + ads[int(tempName[5])] + ads[int(tempName[6])]

    # # crop车牌的左上角和右下角坐标
    # item=item+1
    # print(item)
    # [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
    # x = (leftUp[0] + rightDown[0])/(2*720)
    # y = (leftUp[1] + rightDown[1])/(2*1160)
    # w = (rightDown[0] - leftUp[0])/720
    # h = (rightDown[1] - leftUp[1])/1160
    # list_image.append(str(0) + ' ' +str(x)+' ' +str(y)+' ' +str(w)+' ' +str(h) + '\n')  # 在文件名后加标签

    # new_obj_name = img_name.split('/')[1].rsplit('.', 1)[0]
    # print(new_obj_name)
    # 准备写入
    txt.writelines("ccpd_green/"+str(img_name))
    txt.write('\n')
txt.close()


