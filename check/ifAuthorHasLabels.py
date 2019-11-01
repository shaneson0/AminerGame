from utils import settings
from utils.data_utils import load_json
from utils.cache import LMDBClient
from utils import data_utils


# TrainAuthor = load_json(settings.TRAIN_PUB_DIR, 'train_author.json')
# TrainPub = load_json(settings.TRAIN_PUB_DIR, 'train_pub.json')
#
#
#
# AllPaperNumber = 0
# cnt = 0
# for name in TrainAuthor.keys():
#     Author = TrainAuthor[name]
#     for pid in Author.keys():
#        if TrainPub.__contains__(pid) :
#            print ("%s not in TrainPub"%(pid))
#            cnt += 1
#
#        AllPaperNumber += 1
#
# print ("total number of paper not in TrainPub is: %d"%(cnt))
# print ("total number paper is: %d"%(AllPaperNumber))


# THiFkqbz
# THiFkqbz
# DgeXbWSs

pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')

print (pubs_dict["THiFkqbz"])

# LMDB_AUTHOR_FEATURE = "pub_authors.feature"
# lc_feature = LMDBClient(LMDB_AUTHOR_FEATURE)
# LMDB_AUTHOR_FEATUREs = ['t6jN3XIV', 'k7NZqkRA', 'bZBIUP8E', 'Ls8NZELT', 'qBfjh4lN', 'kzAXIhPb', 'o5Wt9Axr', 'THiFkqbz', 'AdlOinpo', 'ChSQ3usR', '4s927IWa', 'CPS37vxL', '8p2XLcDw', 'O0amz6aO', 'xDqJo9Fn', 'kd8SBHaL', 'QSDG3Qfd', 'RpLlfZKr', 'AujrfYF3', 'lAClLM7i', 'dp0z6PZC', 'Mj9mci7O', 'guzywp6n', 'FFdZI4o5', 'A0f1tMGe', 'FvvSR7EW', 'EkvtYbYG', 'qNznDMGH', 'pwTNjUG7', 'k2kgQJTd', '9RVR6ljL', '6gkbpsWt', 'Y8iyUctH', 'NdsAHdAx', 'XKiEX4VC', 'MazI0I67', 'mHs2sRLd', 'Njl0pi3d', 'rNrepDvl', 'SuQoez3Y', 'THiFkqbz', 'goerVEVC', 'E9fSUI82', 'Xr996urG', 'JkqKYwRk', '291BhSu2', 'AzRIbXzN', '1nFUfkPH', 'ch9gMgyZ', '0fGxKUC9', 'GLwm7cBP', 'KljfbHqI', 'pOegPQ92', 'IueP1wfl', 'po9K9OgK', '2Ng7OKAA', 'rKS8WLep', '6kkwC0k4', 'aeK7BQvC', 'XM7A1p19', 'ufzkz0dp', 'ESm9nYr6', '8YqE2L7O', 'EdhDXC9N', 'ZjFHKEjZ', 'GBcAD3QW', '8BMvY51O', 'ZBbg5WIK', 'LQ3CuCFA', 'Yiw70aJg', 'ofO1sFy0', 'GJCEsbyn', 'A1oRsHOi', 'zHVhDxcD', '14Zkjeue', '5PoS4Inz', 'lHAGTXTP', '1vcHKr45', 'pRZU3uWA', 'aocZ8Qfj', '2Qy6ssvl', 'SuIISWwa', 'nte68MQV', 'wKiCqeKZ', 'WxVElzBo', 'l3vzw8Gr', 'gNH2gJLA', '9aLrAKJC', 'ERHkBfma', 'ynOgEclW', 'R6nzUm5k', 'JqxVwiqW', 'eXsEL0yr', 'tz9AthLk', 'tpLdc7a1', 'MVlAu7Dz', 'fnGLfKaZ', 'xtCLb00k', 'nszu7T5P', 'ZYYT4TKe', 'hvmlehNa', 'vcLqbQR8', 'zVT5LsQY', 'JdoiKozD', 'vVtgjjQk', '0v4oGRj7', 'jBbuQB49', 'h4IH4WIv', 'P7LKEc2Y', 'p7LFwadb', 'czGX9oA1', 'YzFfsp8d', '2U5MF7t4', 'SweHnIdI', 'XgCcloot', 'LXfDTjCL', 'qnuTAfGL', 'f1tMfQi3', 'pEcEQP2H', '801M3Llp', 'TseZXgmT', 'UzWSlC93', '0PEPq2dc', 'OJOeAYLQ', 'wVOWBRow', 'DG3lZuJI', 'Y36KfaUM', '7kcbaZGj', 'ytsfZjoM', 'YN7M1lDb', '85DlkVeY', 'tCSKdCRI', 'cdvxOQ3i', 'xePHaPzo', 'x1PGY8mb', 'Q4y6b2nh', 'IRxsOkCb', '4O6Wj2zG', 'ck1hj6ON', 'BHrXxB51', '1PCxf9UI', '3UvkCzqX', 'NNGjGIVw', 'wu1nhpIe', 'sVYI0l1w', 'dfgg1Npe', '7fNmH5Dp', 'WNh9xkuu', 'vTV8bGmY', 'i5WHaYsx', '4DKDajHK', 'geMlDic6', 'RWe0gMm9', 'Nb3muMKs', 'fwKhPWmU', '5YalIO5W', 'JUnM8WWy', 'UgNfoqSV', 'F1TnchLY', 'PvvHtNXr', 'q51RH99H', 'vOK36p6M', 'rFnLFb1Y', 'EKUjardg', 'LAxw5Ag7', 'XdTHD2ze', 'wUfNZMPI', '2ttNcffY', '104UhmaG', 'RyLUEQT4', 'hYoY449h', 'rLVoFTyk', 'wiAnWwX5', 'tWY0hvTg', 'otb9t8No', 'SfAKHo5l', 'oMLatHko', '09QX9u6j', '8V8W8l4D', '3xOZjkko', 'KS8FwZEP', 'sMNUbfH9', 'GhyIv7iP', 'NFxy3PuZ', '0Dg8qUVf', '7Ide1a3N', 'zRKZMxe5', '4aVuudPp', 'wUBvs11a', 'ttAwZU8g', '2vKcfFBY', 'kWqzcwxb', 'AVikvqwo', 'CNvbXipL', 'uAWLgqhc', 'w3fcZS5Z', 'BPNfgz2m', 'n4kUb7bZ', 'T8VFcorF', 'tDTUDMtf', 'eK681TUI', 'ibpI0Lrg', 'RgSIotO0', 'IyeHcmCM', 'Zgj8oeG8', '3XhUgftj', 'sx1u2Vs1', 'JbZ8n90N', 'DEFoATA0', 'TO2jsazZ', 'wyQPeaXb', 'ximrgUbC', 'DgeXbWSs', 'qAeQj9s5', 'gGkYomY3', 'JBdYPnwX', 'N32CUceq', 'JkCglt0Q', 'uPBhpUaO', 'HjBV852R', 'Xv7cfIhx', 'C2EWpIwD', 'GoVHeRQh', 'fg8DhBwR', '2aCIHHU5', 'FIkCyrcx', 'cX0YIdbw', 'qjEwUdNj', 'UHvVoPZC', '1X7gXHpR', 'g2OZRR4x', 'Hav4BqW1', 'jxzb9GwQ', 'WPQXd0UP', 'cGs76TML', 'lOWsLuAC', 'uGvskX9J', 'XpbJ6Rwn', 'xHfS5eHp', 'iFsJbDSj', 'ihOKHOfP', 'UZBZ818a', 'pPHQGpBp', 'JrE4YLXZ', 'iOzV6DMX', 'yO5DWKe5', '07ScZu07', '11QECz1v', 'InFlGaqs', 'Yht8tihW', 'vHxF08UF', 'BPu5OQO6', 'VgH0Gszn', 'dHMLk42K', 'nDg0vEXu', 'jES8OKDG', '6nE5cnxd', '3acsVhwq', 'KCO6pfUI', 'IotLF3NW', 'ctegNC2u', 'Dk08lwoH', 'jgVz8jKd', 'sw84B8L6', '2ZbWgqIc', 'oWtl4dk4', 'AS3afiZa', 'uMRX0d0O', 'PmeE6a9r', '9cXDbdSE', '4OHb0w9P', 'yS2wsmBR', 'e1Cw7Zfx', 'gi872YSq', 'uOI4wHTV', 'BtcKKXnL', 'D1XJB6Y6', 'Yk24LDVu', '5Hpy1gqI', 'oK3fQ9Hq', 'pNSrlDdC', 'IGBDzOTs']
# for pid in LMDB_AUTHOR_FEATUREs:
#     if lc_feature.get(pid) is None:
#         print (pid)
    # print(lc_feature.get(pid))


