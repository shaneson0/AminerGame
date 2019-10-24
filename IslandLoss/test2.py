from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/tmp/mnist', reshape=False)
batch_images, batch_labels = mnist.train.next_batch(128)

print (batch_labels)
print (set(batch_labels))
