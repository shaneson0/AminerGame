
from IslandLoss.IslandLossModel import CenterLossModel
from IslandLoss.prepareTrainData import prepareData
import tensorflow as tf


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


TrainX, TrainY, TestX, TestY, NumberOfClass = prepareData()
model = CenterLossModel(alpha=0.5, num_classes=NumberOfClass)

# TrainX = list(chunks(TrainX, 1500))
# TrainY = list(chunks(TrainY, 1500))
# x_batch, y_batch = get_Batch(TrainX, TrainY, 1000)
# print (TrainX, TrainY)
model.tarin([TrainX[0]], [TrainY[0]])






