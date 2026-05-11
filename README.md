# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Developing a Convolutional Deep Neural Network (CNN) for Image Classification

Image classification is a fundamental problem in computer vision where the goal is to assign a label to an image based on its visual content. Traditional machine learning methods struggle with high-dimensional image data and fail to capture spatial features effectively.

The objective of this project is to develop a Convolutional Neural Network (CNN) that can automatically extract features from images and accurately classify them into predefined categories.

## Neural Network Model
A Neural Network Model is a computational model inspired by the human brain, used in machine learning and deep learning to recognize patterns, learn from data, and make predictions.
<img width="998" height="698" alt="image" src="https://github.com/user-attachments/assets/a757df16-cd3e-4a0a-99c3-8ac7e249f2af" />

## DESIGN STEPS 
1. Load and Preprocess Data
2. Get the shape of the first image in the training dataset
3. Get the shape of the first image in the test dataset
4. Train the Model
5. Test the Model
6. Predict on a Single Image
7. Display the image


## PROGRAM

### Name: PRAVEEN RAJ R

### Register Number: 212224230207

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name:PRAVEEN RAJ R')
        print('Register Number: 212224230207')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

### OUTPUT

## Training Loss per Epoch
<img width="409" height="306" alt="image" src="https://github.com/user-attachments/assets/37574681-3379-4985-8ed1-fbbf38766fad" />

## Confusion Matrix
<img width="923" height="803" alt="image" src="https://github.com/user-attachments/assets/1390b1be-e146-4666-af0f-0d015247ff2b" />


## Classification Report
<img width="599" height="404" alt="image" src="https://github.com/user-attachments/assets/1d478996-9a33-4344-ac9f-884e158a37e4" />


### New Sample Data Prediction
<img width="528" height="568" alt="image" src="https://github.com/user-attachments/assets/79b054cd-d4c2-41cd-8251-138febc87c1e" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
