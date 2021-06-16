"""Just a quick script to show functionality of opencv."""
import numpy as np
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom as minidom

# ==============================================
# Load up the classifier using frontal face data
# ==============================================
# frontalface_location = './opencvdata/haarcacade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(frontalface_location)
#eye_cascade = cv2.CascadeClassifier('./opencvdata/haarcascade_eye.xml')
# =====================================
# Assign the source and the output file
# =====================================
# Use VideoCapture(0) if you want to use webcam
# cap = cv2.VideoCapture('./videos/ttd-lgbt.mp4')
# out = cv2.VideoWriter('lgbt2.mp4', cv2.cv.CV_FOURCC('X', '2', '6', '4'), 30, (1280, 720))
# ===========================================
# Main loop
# - Iterate through each frame of video
# - Detect face
# - And draw a rectangle around detected face
# ===========================================

import os
def file_name(file_dir):
  # nc=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
  for root, dirs, files in os.walk(file_dir):
    # print(root) #当前目录路径
    # print(dirs) #当前路径下所有子目录
    # print(files) #当前路径下所有非目录子文件
    for file in files:
        # print(file[:-4])
        # writeXML(nc,file[:-4])
        file_image="./images/"+file[:-4]+".jpg"
        if os.path.exists(file_image):
            with open("train.txt", "a") as f:
                f.write("./annotations/"file+" "+file_image)
            


""
# 输入xml文件名称,所有种类
# 返回字典类型种类  类型:(类型:(xmax xmin ymax ymin))
""
# def readXML(nc, filename):
#         domTree = parse("../Annotations/"+filename+".xml")
#         img=cv2.imread(filename+".jpg")
#         # 文档根元素
#         rootNode = domTree.documentElement
#         # print(rootNode.nodeName)
#         elements = rootNode.getElementsByTagName('object')
#         for element in elements:
#             for node in element.childNodes:
#                 # 通过nodeName判断是否是文本
#                 if node.nodeName=='name':
#                     # text = node.data.replace('\n', '')
#                     if node.childNodes[0].nodeValue in nc:
#                         # position_node=element.childNodes[8]
#                         xmin=int(element.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
#                         xmax=int(element.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
#                         ymin = int(element.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
#                         ymax = int(element.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
#                         # print(xmin,type(xmax),ymin,ymax)
#                         cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(255, 255, 255),2)
#                         cv2.imwrite("../save/"+filename+".jpg",img)
#                         # print(element.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
#
#
#                     # print(node.nodeName,node.childNodes[0].nodeValue,node.nodeType)
#                     # if node.nodeName in nc:
#                     #     # 用data属性获取文本内容
#                     #     text = node.data.replace('\n', '')
#                     #     # 这里的文本需要特殊处理一下，会有多余的'\n'
#                     #     print(text)
#
#             # print(element)
#
#         print("   ")
#         # class_name = rootNode.getElementsByTagName("name")
#         # print(class_name[0].txt)
# def readTXT(filename):
#     label_num=[0,0,0,0,0,0,0,0,0]
#     file = open(filename)
#     while 1:
#         line = file.readline()
#         labels=line.split(" ")
#         # for label in labels:
#         for i in range(len(labels)-1):
#             parameters=labels[i+1].split(",")
#             label_num[int(parameters[4])]+=1
#         if not line:
#             break
#         pass  # do something
#     file.close()
#     print(label_num)
# def writeXML(parameters, filename):
#     dom = minidom.getDOMImplementation().createDocument(None, 'annotation', None)
#     root = dom.documentElement
#     for i in range(5):
#         element = dom.createElement('Name')
#         element.appendChild(dom.createTextNode('default'))
#         element.setAttribute('age', str(i))
#         root.appendChild(element)
#     with open(filename+'default.xml', 'w', encoding='utf-8') as f:
#         dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')


# for nameclass in nc:
    #
	# # 所有顾客
	# customers = rootNode.getElementsByTagName("customer")
	# print("****所有顾客信息****")
	# for customer in customers:
	# 	if customer.hasAttribute("ID"):
	# 		print("ID:", customer.getAttribute("ID"))
	# 		# name 元素
	# 		name = customer.getElementsByTagName("name")[0]
	# 		print(name.nodeName, ":", name.childNodes[0].data)
	# 		# phone 元素
	# 		phone = customer.getElementsByTagName("phone")[0]
	# 		print(phone.nodeName, ":", phone.childNodes[0].data)
	# 		# comments 元素
	# 		comments = customer.getElementsByTagName("comments")[0]
	# 		print(comments.nodeName, ":", comments.childNodes[0].data)


if __name__ == '__main__':
    file_name('./annotations')
    # readTXT("sogou_test7898.txt")

# counter = 0
# while True:
#     counter += 1
#     ret, frame = cap.read()
#     # Change the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Apply classifier to find face
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     # Draw a rectangle
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#     # Draw the frame with the rectangle back
#     out.write(frame)
#     # cv2.imshow('frame', frame)
#     # if counter % 3 == 0:
#     #     cv2.imshow('frame', frame)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     if ret != True:
#         break
# # Clean up
# cap.release()
# out.release()
# cv2.destroyAllWindows()
