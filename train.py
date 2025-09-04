from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2
import torch
from PIL import Image, ImageOps
import pillow_avif
from model import Model
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error
from customDataSet import CustomDataSet
from torch import nn

if __name__ == "__main__":  
    #load the resnet50 model
    fullModel = resnet50(weights=ResNet50_Weights.DEFAULT)
    #get everyhting but the last softmax layer
    usableModel = torch.nn.Sequential(*list(fullModel.children())[:-1])
    #set it to eval mode bc we aren't training the CNN just the linear
    usableModel.eval()
    model = Model()
    poopTuple = []
    criterion = nn.MSELoss()

    df = pd.read_csv("./data/carbonCredits.csv")
    values = torch.tensor(df["carbonCredits"].values, dtype=torch.float)

    for i in range(2):
        image = Image.open(f"data/modified_pig{i+1}.avif")
        transforms = v2.Compose([
            v2.ToImage(),
        ])
        img = (transforms(image).float()/256).unsqueeze(0)
        embedding = usableModel(img)
        embedding = embedding.view(embedding.size(0), -1)
        poopTuple.append(embedding)

    concatShit = torch.cat(poopTuple, 0)
    concatShit = torch.flatten(concatShit).unsqueeze(0)
    customDS = CustomDataSet(concatShit, values)

    dataloader = DataLoader(customDS, batch_size=1)
    optimizer = torch.optim.SGD(model.parameters())
    model.train()
    for xBatch, yBatch in dataloader:
        optimizer.zero_grad()
        y_pred = model(xBatch)
        y_pred = torch.flatten(y_pred)
        loss = torch.sqrt(criterion(y_pred, yBatch))
        loss.backward()
        optimizer.step()
    








