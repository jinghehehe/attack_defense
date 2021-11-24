"""
Dump the adversarial images info(absolute path) of each target model into json
Results stored in ADV.
"""
import os
import json

models = ['vgg', 'res', 'dense', 'trans']

for model in models:
    train_imgs = './' + model + '_train/'
    val_imgs = './' + model + '_val/'
    if not os.path.exists(train_imgs):
        os.mkdir(train_imgs)
    if not os.path.exists(val_imgs):
        os.mkdir(val_imgs)

    train_file = os.listdir(train_imgs)
    val_file = os.listdir(val_imgs)

    time_train = [os.stat(train_imgs + img).st_mtime for img in train_file]
    time_val = [os.stat(val_imgs + img).st_mtime for img in val_file]

    train_meta = []
    val_meta = []
    img_dir = '/data/private_data/jf_private/yizhi_data/LP_AD/CW/'

    for file_name in train_file:
        if 'real' in file_name:
            continue
        else:
            img_path = img_dir +  model + '_train/' + file_name
            dict = {'img_name': img_path}
            train_meta.append(dict)

    for file_name in val_file:
        if 'adv' in file_name:
            continue
        else:
            img_path = img_dir + model + '_val/' + file_name
            dict = {'img_name': img_path}
            val_meta.append(dict)


    print(model + ":", len(train_meta),'/ 237900;  ', len(val_meta),'/ 65312')
    if len(time_train) > 0:
        train_cost = max(time_train) - min(time_train)
        print('  '+model + ' train expected:  %.2f h ' % (train_cost*(237900-len(train_meta))/(len(train_meta)*3600)) )
    if len(time_val) > 0:
        val_cost = max(time_val) - min(time_val)
        print('  '+model + ' val expected:  %.2f h' % (val_cost * (65312-len(val_meta)) / (len(val_meta) * 3600)) )

    with open('ADV/' + model + '_train.json', 'w') as file:
        json.dump(train_meta, file)

    with open('ADV/' + model + '_val.json', 'w') as file:
        json.dump(val_meta, file)
