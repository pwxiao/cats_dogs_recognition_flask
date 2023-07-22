from flask import Flask,request,render_template
import os
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable

app = Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'predict')

def predict(image_path):
    model = torch.load("model/model.pth")
    img = Image.open(image_path)
    data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor()])  
    img = data_transform(img)
    x = []
    for i in range(16):
        x.append(img)

    x = torch.stack(x, dim=0)
    x = Variable(x.cuda())
    y = model(x)
    y = y[0]
    if y[0] < y[1]:
        return "狗"
    else:
        return "猫"


@app.route('/')
def main():
    return render_template("index.html")



@app.route('/predict', methods=['POST'])
def upload():
    # 获取上传的文件
    file = request.files['file']

    # 保存上传的文件
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # 调用处理函数处理图像
    prediction = predict(file_path)

    # 返回预测结果
    return prediction


if __name__ == '__main__':
   
    app.run(debug=True)