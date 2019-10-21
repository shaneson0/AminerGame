
from IslandLoss.IslandLossModel import CenterLossModel
from IslandLoss.prepareTrainData import prepareData

TrainX, TrainY, TestX, TestY = prepareData()
model = CenterLossModel()
model.tarin(TrainX, TrainY)

