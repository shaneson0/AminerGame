
from IslandLoss.IslandLossModel import CenterLossModel
from IslandLoss.prepareTrainData import prepareData

TrainX, TrainY, TestX, TestY, NumberOfClass = prepareData()
model = CenterLossModel(alpha=0.5, num_classes=NumberOfClass)
model.tarin(TrainX, TrainY)

