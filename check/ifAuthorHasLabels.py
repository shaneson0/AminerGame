from utils import settings
from utils.data_utils import load_json


TrainAuthor = load_json(settings.TRAIN_PUB_DIR, 'train_author.json')
TrainPub = load_json(settings.TRAIN_PUB_DIR, 'train_pub.json')


AllPaperNumber = 0
cnt = 0
for name in TrainAuthor.keys():
    Author = TrainAuthor[name]
    for pid in Author.keys():
       if pid not in TrainPub:
           print ("%s not in TrainPub"%(pid))
           cnt += 1

       AllPaperNumber += 1

print ("total number of paper not in TrainPub is: %d"%(cnt))
print ("total number paper is: %d"%(AllPaperNumber))


