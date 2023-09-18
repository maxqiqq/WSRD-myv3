# TODO:当前py工程父目录
# import os
# print(os.getcwd())
# import cv2


# TODO:cv2检测channel
# image_paths = ["C:/Users/maxiqq/Desktop/2b13.png", "C:/Users/maxiqq/Desktop/1a11.jpg"]
# for image_path in image_paths:
#     # 读取图像
#     img = cv2.imread(image_path)
#     # 查看通道数
#     print(f"{image_path}的通道数是：{img.shape[2]}")


# TODO:PIL cv2通道检测不同验证
# from PIL import Image
# import cv2
#
# inp_data = Image.open('D:/PyCharm/PycharmProjects/WSRD-DNSR/dataset/train/train_A/2b12.png')
# print(inp_data.mode)
# # 转换图像
# # inp_data_rgb = inp_data.convert('RGB')
# # print(inp_data_rgb.mode)
#
# image_path = 'D:/PyCharm/PycharmProjects/WSRD-DNSR/dataset/train/train_A/2b12.png'
# image = cv2.imread(image_path)
# print(f"{image_path}的通道数是：{image.shape[2]}")


# TODO:转换文件夹所有image通道数 并 验证
# import os
# from PIL import Image
#
# input_dir = r'D:\PyCharm\PycharmProjects\WSRD-DNSR\dataset\train\train_A_4'
# output_dir = r'D:\PyCharm\PycharmProjects\WSRD-DNSR\dataset\train\train_A'
#
# for filename in os.listdir(input_dir):
#     # 只处理.png文件
#     if filename.endswith('.png'):
#         # 打开图像
#         img = Image.open(os.path.join(input_dir, filename))
#         # 将图像转换为RGB
#         img_rgb = img.convert('RGB')
#         # 保存新的RGB图像
#         img_rgb.save(os.path.join(output_dir, filename))
# print("所有图像已成功转换为RGB并保存在指定的输出目录中。")

# inp_data = Image.open('C:/Users/maxiqq/Desktop/1a11.png')
# print(inp_data.mode)
# inp_data = Image.open('C:/Users/maxiqq/Desktop/2b11.png')
# print(inp_data.mode)






