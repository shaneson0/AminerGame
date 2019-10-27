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

def encode_labels2(labels):
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
    TestLabelDict, TestLabelNumberofCluss = preprocessTestLabels()
    LabelDict, numberofCluss = preprocessLabels()
    TrainPids = np.array(list(LabelDict.keys()))
    AllPids = np.array(TrainPids)

    TrainPids, ValidPids = train_test_split(AllPids, stratify=list(LabelDict.values()), test_size=0.1, random_state=42)
    TestPids = np.array(list(TestLabelDict.keys()))

    # TrainPids, ValidPids = train_test_split(TrainPids, test_size=0.1, random_state=42)


    LMDB_NAME_EMB = "publication.emb.weighted"
    lc_emb = LMDBClient(LMDB_NAME_EMB)

    AllX = []
    Ally = []
    TrainX = []
    TrainY = []
    ValidX = []
    ValidY = []


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

    for pid in ValidPids:
        emb = lc_emb.get(pid)
        label = LabelDict[pid]
        # print ("pid: ", pid, ", label: ", label, ', emb: ', emb)
        if emb is None:
            continue
        AllX.append(emb)
        ValidX.append(emb)
        ValidY.append(label)
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

    return np.array(TrainX), np.array(TrainY), np.array(ValidX), np.array(ValidY), numberofCluss, AllX, Ally, list(TrainPids) + list(ValidPids), np.array(TestX), np.array(TestY)

def preprocessTestLabels():
    LabelDict, numberofCluss = genPublicationLabel()
    CntList = np.zeros(numberofCluss)
    for key in LabelDict:
        CntList[LabelDict[key]] += 1


    TestLabel = []
    ValidLabel = []
    for label in range(numberofCluss):
        if CntList[label] < 300:
            ValidLabel.append(label)

    # 3724
    # 258
    TestLabelDict = {}
    NewLabelDict = {}
    for key in LabelDict:
        label = LabelDict[key]
        if label in ValidLabel:
            NewLabelDict[key] = label

        if label in TestLabel:
            TestLabelDict[key] = label


    classes_dict, numberofCluss = encode_labels2(list(NewLabelDict.values()))

    for key in NewLabelDict:
        oldLabel = NewLabelDict[key]
        NewLabelDict[key] = classes_dict[oldLabel]

    return NewLabelDict, numberofCluss

def preprocessLabels():
    LabelDict, numberofCluss = genPublicationLabel()
    CntList = np.zeros(numberofCluss)
    for key in LabelDict:
        CntList[LabelDict[key]] += 1


    TestLabel = []
    ValidLabel = []
    for label in range(numberofCluss):
        if CntList[label] > 300:
            ValidLabel.append(label)

    # 3724
    # 258
    TestLabelDict = {}
    NewLabelDict = {}
    for key in LabelDict:
        label = LabelDict[key]
        if label in ValidLabel:
            NewLabelDict[key] = label

        if label in TestLabel:
            TestLabelDict[key] = label


    classes_dict, numberofCluss = encode_labels2(list(NewLabelDict.values()))

    for key in NewLabelDict:
        oldLabel = NewLabelDict[key]
        NewLabelDict[key] = classes_dict[oldLabel]

    return NewLabelDict, numberofCluss


    # CntList = np.array(CntList)
    # CntList = CntList[CntList > 10]
    #
    # print (list(CntList))
    # print (len(list(CntList)))
    # print (len(list(set(CntList))))



if __name__ == '__main__':
    TrainX, TrainY, ValidX, ValidY, NumberOfClass, AllX, Ally = prepareData()
    print (NumberOfClass)
    print (TrainX)
    print (TrainY)
    print (len(set(TrainY)))
    print (len(set(ValidY)))



















