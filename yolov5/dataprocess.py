"""
	创建images和labels的.txt文件
"""
import os
import random
import xml.etree.ElementTree as ET
import shutil
import glob
import tqdm
import concurrent.futures


#img_root = '/home/hanz/workspace/HWT2/images'#原始图片
#lab_root = "/home/hanz/workspace/HWT2/Annotations/"#原始标签
#img_save_root = '/home/hanz/workspace/HWT2/image_path'#保存处理后的图片
#lab_save_root = '/home/hanz/workspace/HWT2/labels'#保存处理后的标签
#error_label_root = '/home/hanz/workspace/HWT2/error_labels'    # 记录错误标记

img_root = "/media/data/qkl/25000_random/task1/images/"#原始图片
lab_root = "/media/data/qkl/25000_random/task1/Annotations/"#原始标签
img_save_root = "/media/data/qkl/25000_random/task1/image_path"#保存处理后的图片
lab_save_root = "/media/data/qkl/25000_random/task1/labels"#保存处理后的标签
error_label_root = "/media/data/qkl/25000_random/task1/error_labels"    # 记录错误标记

#img_root = "/home/hanz/workspace/HWTMLB4/images"
#lab_root = "/home/hanz/workspace/HWTMLB4/Annotations"
#img_save_root = "/home/hanz/workspace/HWTMLB4/image_path"
#lab_save_root = "/home/hanz/workspace/HWTMLB4/labels"
#error_label_root = "/home/hanz/workspace/HWTMLB4/error_labels"    # 记录错误标记

for path in img_save_root,lab_save_root,error_label_root:
    os.makedirs(path, exist_ok=True)
classes = ['others']

#归一化坐标
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1 #边界框的左上角和右下角的 x 坐标。
    y = (box[2] + box[3]) / 2.0 - 1 #边界框的左上角和右下角的 y 坐标。
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return x, y, w, h
#将一组 XML 文件中的物体边界框信息提取出来，并转换为指定格式的文本文件
def convert_xml(xml_files, pbar):
    for xml_file in xml_files:
        pbar.update(1)
        filename = os.path.basename(xml_file)[:-4]
        if (os.path.exists(lab_save_root + '/%s.txt' % filename)):
            continue#检查是否已经存在与当前 XML 文件对应的输出文本文件，如果存在则跳过处理
        out_file = open(lab_save_root + '/%s.txt' % filename, 'w')#打开一个新的文本文件以写入处理后的边界框信息
        tree = ET.parse(xml_file)  # 读取文档
        root = tree.getroot()  # 获取根节点
        size = root.find('size')#在根节点中查找 size 标签，获取图像的宽度和高度信息
        w = float(size.find('width').text)  # .text 获取内容
        h = float(size.find('height').text)
        if w <= 0 or h <= 0:
            #print('size标签有误: ', xml_file)
            error_label.write(xml_file + ' --size标签有误\n')
            continue                            
        for obj in root.iter('object'):  # root.iter('object')递归查找所有子节点
            for emt1 in obj:
                for emt2 in emt1:
                    cls = emt2.tag#获取子元素的标签名称，通常表示物体的类别
                    if cls not in classes:
                        cls = 'others'
                    cls_id = classes.index(cls)#获取类别在 classes 列表中的索引，用于标识物体类别。
                    bndbox = emt2.find('bndbox')#查找子元素中的 bndbox 标签，该标签包含物体的边界框信息
                    xmin = float(bndbox.find('xmin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymin = float(bndbox.find('ymin').text)
                    ymax = float(bndbox.find('ymax').text)
                    if xmin < 0 or xmax > w or ymin < 0 or xmin > xmax or ymin > ymax:
                        #print('bndbox标签有误--%s: '%cls, xml_file)
                        error_label.write(xml_file + ' --%s 的bndbox标签有误\n'%cls)
                        continue
                    if ymax > h:
                        ymax = h
                    box = [xmin, xmax, ymin, ymax]#将边界框的坐标存储在列表 box 中
                    yolobox = convert([w, h], box)
                    out_file.write(str(cls_id) + ' ' + ' '.join([str(a) for a in yolobox]) + '\n')#将处理后的边界框信息写入输出文件，格式为：类别索引 + 归一化后的边界框坐标
        out_file.close()

#并行处理一组 XML 文件，将它们转换为特定格式的文本文件
def run_convert_xml():
    xml_filepath = lab_root + '/*.xml'
    xml_files = glob.glob(xml_filepath)
    xml_nums = len(xml_files)
    print("Total Xml Nums: %d" % xml_nums)
    worker_nums = 8  # 使用线程数
    print("worker_nums = %d\nstart convert..." % worker_nums)
    set_len = int(xml_nums / worker_nums)  # 每组处理的数量
    start_index = [x * set_len for x in range(worker_nums)]
    progressbars = [tqdm.tqdm(total = set_len) for _ in range(worker_nums)]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_xml, xml_files[x : x + set_len], pbar) for x, pbar in zip(start_index, progressbars)]
        concurrent.futures.wait(futures) 
        for pbar in progressbars:
            pbar.close()
    if (xml_nums - worker_nums * set_len):
        convert_xml(xml_files[worker_nums * set_len : ], tqdm.tqdm(total = xml_nums - worker_nums * set_len))


#shutil.rmtree(lab_save_root)  # 清空labels
#os.mkdir(lab_save_root)
error_label = open(error_label_root + '/error_label.txt','w')
error_label.truncate(0)  # 清空文件内容
run_convert_xml()
error_label.close()
print('\n\033[33m--------------------------------------labels准备完毕--------------------------------------')

labels = os.listdir(lab_save_root)
labels = sorted(labels)
setsize = 2
num = int(len(labels) / setsize)
with open(error_label_root + '/error_label.txt') as file:
    lines = file.readlines()
    error_num = len(lines)
print('成功读取了 %d 个标签文件,保存在 \'%s\' 中'%(num*setsize, lab_save_root))
print('有 %d 个标记有误，记录在 \'%s\' 中\033[0m'%(error_num,error_label_root + '/error_label.txt'))

lst = range(num)
tv_percent = 0.9  # 训练:验证:测试=6:2:2
tr_percent = 0.8  
num_tv = round(num * tv_percent)  # 训练集和验证集数目
num_tr = round(num * tr_percent)
tv = random.sample(lst, num_tv)
tr = random.sample(tv, num_tr)
image_num = 0
with open(img_save_root + '/train.txt', 'w') as train,\
     open(img_save_root + '/val.txt', 'w') as val,\
     open(img_save_root + '/test.txt', 'w') as test:
    for i in lst:
        for j in range(setsize):
            image_path = str(img_root) +'/'+ labels[setsize*i+j][:-3] + 'png' + '\n'
            if i in tv:
                if i in tr:
                    train.write(image_path)
                else:
                    val.write(image_path)
            else:
                test.write(image_path)
            image_num += 1
                
print('\033[33m--------------------------------------images准备完毕--------------------------------------')
print('成功读取了 %d 个图片路径,保存在 \'%s\' 中'%(image_num,img_save_root))
print('训练:验证:测试 = %d:%d:%d\033[0m'%(tr_percent*10,tv_percent*10-tr_percent*10,10-tv_percent*10))


