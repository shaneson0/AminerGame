
from IslandLoss.IslandLossModel import CenterLossModel
from IslandLoss.prepareTrainData import prepareData
import tensorflow as tf


def get_Batch(data, label, batch_size):
    # print(data.shape, label.shape)
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32 )
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
    return x_batch, y_batch


TrainX, TrainY, TestX, TestY, NumberOfClass = prepareData()
model = CenterLossModel(alpha=0.5, num_classes=NumberOfClass)


print (TrainX.shape, TrainY.shape)
model.tarin(TrainX, TrainY)






