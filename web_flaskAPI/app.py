from flask import Flask, request, render_template

import torch
from torchvision import transforms
from PIL import Image

# Load model
class CNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.batch1 = torch.nn.BatchNorm2d(8)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.batch2 = torch.nn.BatchNorm2d(16)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batch3 = torch.nn.BatchNorm2d(32)
        self.act3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch4 = torch.nn.BatchNorm2d(64)
        self.act4 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batch5 = torch.nn.BatchNorm2d(128)
        self.act5 = torch.nn.ReLU()
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2)

        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.batch1(self.conv1(x))))
        x = self.pool2(self.act2(self.batch2(self.conv2(x))))
        x = self.pool3(self.act3(self.batch3(self.conv3(x))))
        x = self.pool4(self.act4(self.batch4(self.conv4(x))))
        x = self.pool5(self.act5(self.batch5(self.conv5(x))))
        x = self.flatten(x)
        return torch.nn.functional.log_softmax(self.fc(x), dim=1)

# Initialize Flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load the pre-trained model
device = torch.device("cpu")
model = CNN(num_classes=10).to(device)
model.load_state_dict(torch.load("Kali_Turing_CNN.pt", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route('/') 
def home(): 
    title = "Home Page"  # Set the title for the current page 
    return render_template('index.html', title=title) 

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('predict.html', error="No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('predict.html', error="No file selected!")

    try:
        # Load and process the image
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Class labels
        class_labels = [
            "Tomato__Bacterial_spot",
            "Tomato__Early_blight",
            "Tomato__Late_blight",
            "Tomato__Leaf_Mold",
            "Tomato__Septoria_leaf_spot",
            "Tomato__Spider_mites Two-spotted_spider_mite",
            "Tomato__Target_Spot",
            "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato__Tomato_mosaic_virus",
            "Tomato__healthy"
        ]
        predicted_class = class_labels[predicted.item()]

        # Render the result page with the predicted class
        return render_template('predict.html', prediction=predicted_class)
    
    except Exception as e:
        return render_template('predict.html', error=f"Error: {str(e)}")
if __name__ == '__main__':
    app.run(debug=True)

