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

def encode_sna_labels(sna_author):
    labels = []
    for aid in sna_author.keys():
        labels.append(aid)
    classes = set(labels)
    numberofCluss = len(list(set(labels)))
    classes_dict = {c: i for i, c in enumerate(classes)}
    return classes_dict, numberofCluss

def encode_labels(train_author, sna_valid_pub):
    labels = []
    for name in train_author.keys():
        for aid in train_author[name].keys():
            labels.append(aid)

    for aid in sna_valid_pub.keys():
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

def genSNAData():
    Label = {}
    with open(join(settings.SNA_PUB_DIR, "sna_valid_author_raw.json"), "r") as fp:
        SNAAuthor = json.load(fp)
        fp.close()
    classes_dict, numberofCluss = encode_sna_labels(SNAAuthor)

    for aid in SNAAuthor.keys():
        for pid in SNAAuthor[aid]:
            Label[pid] = classes_dict[aid]

    Newlabel = {}
    CntList = np.zeros(numberofCluss)
    CntLabel = np.zeros(numberofCluss)
    for key in Label:
        CntList[Label[key]] += 1

    for key in Label:
        if CntLabel[Label[key]] > 100 or CntList[Label[key]] < 100:
            continue
        CntLabel[Label[key]] += 1
        Newlabel[key] = Label[key]


    return Newlabel, len(list(set(Newlabel.values())))

def genSNALabel():
    Label = {}
    with open(join(settings.SNA_PUB_DIR, "sna_valid_author_raw.json"), "r") as fp:
        sna_valid_pub = json.load(fp)
        fp.close()
    classes_dict, numberofCluss = encode_labels({}, sna_valid_pub)

    for aid in sna_valid_pub.keys():
        for pid in sna_valid_pub[aid]:
            Label[pid] = classes_dict[aid]

    return Label, numberofCluss

def genPublicationLabel():
    Label = {}
    with open(join(settings.TRAIN_PUB_DIR, "train_author.json"), "r") as fp:
        train_author = json.load(fp)
        fp.close()

    with open(join(settings.SNA_PUB_DIR, "sna_valid_author_raw.json"), "r") as fp:
        sna_valid_pub = json.load(fp)
        fp.close()

    classes_dict, numberofCluss = encode_labels(train_author, sna_valid_pub)

    for name in train_author.keys():
        for aid in train_author[name].keys():
            for pid in train_author[name][aid]:
                Label[pid] = classes_dict[aid]

    for aid in sna_valid_pub.keys():
        for pid in sna_valid_pub[aid]:
            Label[pid] = classes_dict[aid]

    return Label, numberofCluss



def prepareData(type='train'):
    # SNALabelDict, numberofCluss = preprocessSNALabels()
    # TestLabelDict, TestLabelNumberofCluss = preprocessTestLabels()
    if type == 'train':
        LabelDict, numberofCluss = preprocessLabels()
    else:
        LabelDict, numberofCluss = preprocessSNALabels()

    print ("LabelDict : ", LabelDict)


    TrainPids = np.array(list(LabelDict.keys()))
    AllPids = np.array(TrainPids)
    print ("AllPids : ", list(AllPids))

    if type == 'train':
        TrainPids, ValidPids = train_test_split(AllPids, stratify=list(LabelDict.values()), random_state=42)
    else:
        TrainPids, ValidPids = train_test_split(AllPids, random_state=42)

    # TrainPids, ValidPids = train_test_split(TrainPids, test_size=0.1, random_state=42)

    LMDB_NAME_EMB = "publication.emb.weighted"
    lc_emb = LMDBClient(LMDB_NAME_EMB)

    AllX = []
    Ally = []
    TrainX = []
    TrainY = []
    ValidX = []
    ValidY = []
    Allpids = []


    for pid in TrainPids:
        emb = lc_emb.get(pid)
        label = LabelDict[pid]
        # print ("pid: ", pid, ", label: ", label, ', emb: ', emb)
        if emb is None:
            continue
        Allpids.append(pid)
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
        Allpids.append(pid)
        AllX.append(emb)
        ValidX.append(emb)
        ValidY.append(label)
        Ally.append(label)


    return np.array(TrainX), np.array(TrainY), np.array(ValidX), np.array(ValidY), numberofCluss, AllX, Ally, Allpids

def preprocessTestLabels():
    LabelDict, numberofCluss = genPublicationLabel()
    CntList = np.zeros(numberofCluss)
    for key in LabelDict:
        CntList[LabelDict[key]] += 1


    TestLabel = []
    ValidLabel = []
    for label in range(numberofCluss):
        if CntList[label] < 300 and CntList[label] > 250:
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


def preprocessSNALabels():
    LabelDict, numberofCluss = genSNAData()
    CntList = np.zeros(numberofCluss)
    print ("numberofCluss: ", numberofCluss)
    print ("LabelDict values: ", len(list(set(LabelDict.values()))))
    for key in LabelDict:
        CntList[LabelDict[key]] += 1

    TestLabel = []
    ValidLabel = []
    for label in range(numberofCluss):
        if CntList[label] > 100:
            ValidLabel.append(label)

    # 3724
    # 258
    SNALabelDict = {}
    for key in LabelDict:
        label = LabelDict[key]
        if label in ValidLabel:
            SNALabelDict[key] = label


    classes_dict, numberofCluss = encode_labels2(list(SNALabelDict.values()))

    for key in SNALabelDict:
        oldLabel = SNALabelDict[key]
        SNALabelDict[key] = classes_dict[oldLabel]

    return SNALabelDict, numberofCluss

from collections import  defaultdict
def preprocessLabels():
    LabelDict, numberofCluss = genPublicationLabel()
    CntList = np.zeros(numberofCluss)
    for key in LabelDict:
        CntList[LabelDict[key]] += 1


    TestLabel = []
    ValidLabel = []
    for key in LabelDict:
        if CntList[LabelDict[key]] > 300:
            ValidLabel.append(LabelDict[key])

    ValidLabel = list(set(ValidLabel))

        # CntList[LabelDict[key]] += 1
    #
    # for label in range(numberofCluss):
    #     if CntList[label] > 300:
    #         ValidLabel.append(label)

    # 3724
    # 258
    LabelCnt = defaultdict(int)
    TestLabelDict = {}
    NewLabelDict = {}
    for key in LabelDict:
        label = LabelDict[key]
        if LabelCnt[label] > 59:
            continue

        if label in ValidLabel:
            NewLabelDict[key] = label
            LabelCnt[label] += 1

        if label in TestLabel:
            TestLabelDict[key] = label


    classes_dict, numberofCluss = encode_labels2(list(NewLabelDict.values()))

    for key in NewLabelDict:
        oldLabel = NewLabelDict[key]
        NewLabelDict[key] = classes_dict[oldLabel]

    return NewLabelDict, numberofCluss

def preprocessSNALabels():
    LabelDict, numberofCluss = genSNALabel()
    classes_dict, numberofCluss = encode_labels2(list(LabelDict.values()))

    for key in LabelDict:
        oldLabel = LabelDict[key]
        LabelDict[key] = classes_dict[oldLabel]

    return LabelDict, numberofCluss



if __name__ == '__main__':
    genSNAData()
    # TrainX, TrainY, ValidX, ValidY, NumberOfClass, AllX, Ally = prepareData()
    # print (NumberOfClass)
    # print (TrainX)
    # print (TrainY)
    # print (len(set(TrainY)))
    # print (len(set(ValidY)))



















