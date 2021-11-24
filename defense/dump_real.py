"""
Dump the real images info(absolute path) of each target model into json
Results stored in ADV. Only for val set.
"""
import os
import json

models = ['dense']

for model in models:
    val_imgs = './' + model + '_val/'
    if not os.path.exists(val_imgs):
        os.mkdir(val_imgs)

    val_file = os.listdir(val_imgs)
    val_file.sort()
    print(val_file)
    val_meta = []

    img_dir = '/data/private_data/jf_private/yizhi_data/LP_AD/CW/'

    for file_name in val_file:
        if 'adv' in file_name :
            continue
        else:
            #img_path = img_dir + model + '_val/' + file_name
            #d_path = img_path.replace(model, 'dense')
            #if not os.path.exists(d_path):
            dict = {'img_name': img_path}
            val_meta.append(dict)

    print(len(val_meta))
    with open('ADV/' + 'real.json', 'w') as file:
        json.dump(val_meta, file)
