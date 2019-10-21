
from IslandLoss.IslandLossModel import CenterLossModel
from IslandLoss.prepareTrainData import prepareData

TrainX, TrainY, TestX, TestY = prepareData()
model = CenterLossModel(alpha=0.5)
model.tarin(TrainX, TrainY)

