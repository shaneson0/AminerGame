from utils import data_utils
from utils import settings
import codecs
from os.path import join
from datetime import datetime
from utils import feature_utils
from global_.embedding import EmbeddingModel
from utils.cache import LMDBClient
from collections import defaultdict as dd
from sklearn.model_selection import train_test_split
import math
import numpy as np
import json

def encode_labels(train_author):
    labels = []
    for name in train_author.keys():
        for aid in train_author[name].keys():
            labels.append(aid)

    # print ("classes number: ", len(list(set(labels))))
    classes = set(labels)
    numberofCluss = len(list(set(labels)))
    classes_dict = {c: i for i, c in enumerate(classes)}
    return classes_dict, numberofCluss

def genPublicationLabel():
    Label = {}
    with open(join(settings.TRAIN_PUB_DIR, "train_author.json"), "r") as fp:
        train_author = json.load(fp)
        fp.close()
    classes_dict, numberofCluss = encode_labels(train_author)

    for name in train_author.keys():
        for aid in train_author[name].keys():
            for pid in train_author[name][aid]:
                Label[pid] = classes_dict[aid]
    return Label, numberofCluss

def prepareData():
    # prepare Data
    TrainPids = []
    with open(join(settings.TRAIN_PUB_DIR, "train_author.json"), "r") as fp:
        train_author = json.load(fp)
        fp.close()
    for name in train_author.keys():
        for aid in train_author[name]:
            TrainPids = TrainPids + train_author[name][aid]

    TrainPids = np.array(TrainPids)

    TrainPids, TestPids = train_test_split(TrainPids, test_size=0.33, random_state=42)
    LabelDict, numberofCluss = preprocessLabels()

    LMDB_NAME_EMB = "publication.emb.weighted"
    lc_emb = LMDBClient(LMDB_NAME_EMB)

    AllX = []
    Ally = []
    TrainX = []
    TrainY = []
    TestX = []
    TestY = []

    for pid in TrainPids:
        emb = lc_emb.get(pid)
        label = LabelDict[pid]
        # print ("pid: ", pid, ", label: ", label, ', emb: ', emb)
        if emb is None:
            continue
        AllX.append(emb)
        TrainX.append(emb)
        TrainY.append(label)
        Ally.append(label)

    for pid in TestPids:
        emb = lc_emb.get(pid)
        label = LabelDict[pid]
        # print ("pid: ", pid, ", label: ", label, ', emb: ', emb)
        if emb is None:
            continue
        AllX.append(emb)
        TestX.append(emb)
        TestY.append(label)
        Ally.append(label)

    return np.array(TrainX), np.array(TrainY), np.array(TestX), np.array(TestY), numberofCluss, AllX, Ally


def preprocessLabels():
    LabelDict, numberofCluss = genPublicationLabel()
    CntList = np.zeros(numberofCluss)
    for key in LabelDict:
        CntList[LabelDict[key]] += 1

    ValidLabel = []
    for label in range(numberofCluss):
        if CntList[label] > 50:
            ValidLabel.append(label)
    # 3724
    # 258
    NewLabelDict = {}
    for key in LabelDict:
        label = LabelDict[key]
        if label in ValidLabel:
            NewLabelDict[key] = label

    # print (NewLabelDict)
    # print (len(NewLabelDict.keys()))
    # print (len(set(NewLabelDict.values())))
    numberofCluss = len(set(NewLabelDict.values()))

    return NewLabelDict, numberofCluss


    # CntList = np.array(CntList)
    # CntList = CntList[CntList > 10]
    #
    # print (list(CntList))
    # print (len(list(CntList)))
    # print (len(list(set(CntList))))



if __name__ == '__main__':
    preprocessLabels()

###
















