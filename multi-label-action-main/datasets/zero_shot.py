import numpy as np

split_50_train = [] ## 50:50 split
split_50_test = [] ## 50:50 split

with open('datasets/splits/train_50_test_50/THUMOS14/train/split_0.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('datasets/splits/train_50_test_50/THUMOS14/test/split_0.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_50_train.append(files2[:-1])

for files3 in filecontents3:
    split_50_test.append(files3[:-1])

split_50_train_dict = { split_50_train[i] : i for i in sorted(range(len(split_50_train)))}
split_50_test_dict = { split_50_test[i] : i for i in sorted(range(len(split_50_test)))}

split_75_train = [] ## 50:50 split
split_25_test = [] ## 50:50 split

with open('datasets/splits/train_75_test_25/THUMOS14/train/split_0.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('datasets/splits/train_75_test_25/THUMOS14/test/split_0.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_75_train.append(files2[:-1])

for files3 in filecontents3:
    split_25_test.append(files3[:-1])

split_75_train_dict = { split_75_train[i] : i for i in sorted(range(len(split_75_train)))}
split_25_test_dict = { split_25_test[i] : i for i in sorted(range(len(split_25_test)))}


charades_dict = {'Holding some clothes': 0, 'Putting clothes somewhere': 1, 'Taking some clothes from somewhere': 2, 'Throwing clothes somewhere': 3, 'Tidying some clothes': 4, 'Washing some clothes': 5, 'Closing a door': 6, 'Fixing a door': 7, 'Opening a door': 8, 'Putting something on a table': 9, 'Sitting on a table': 10, 'Sitting at a table': 11, 'Tidying up a table': 12, 'Washing a table': 13, 'Working at a table': 14, 'Holding a phone/camera': 15, 'Playing with a phone/camera': 16, 'Putting a phone/camera somewhere': 17, 'Taking a phone/camera from somewhere': 18, 'Talking on a phone/camera': 19, 'Holding a bag': 20, 'Opening a bag': 21, 'Putting a bag somewhere': 22, 'Taking a bag from somewhere': 23, 'Throwing a bag somewhere': 24, 'Closing a book': 25, 'Holding a book': 26, 'Opening a book': 27, 'Putting a book somewhere': 28, 'Smiling at a book': 29, 'Taking a book from somewhere': 30, 'Throwing a book somewhere': 31, 'Watching/Reading/Looking at a book': 32, 'Holding a towel/s': 33, 'Putting a towel/s somewhere': 34, 'Taking a towel/s from somewhere': 35, 'Throwing a towel/s somewhere': 36, 'Tidying up a towel/s': 37, 'Washing something with a towel': 38, 'Closing a box': 39, 'Holding a box': 40, 'Opening a box': 41, 'Putting a box somewhere': 42, 'Taking a box from somewhere': 43, 'Taking something from a box': 44, 'Throwing a box somewhere': 45, 'Closing a laptop': 46, 'Holding a laptop': 47, 'Opening a laptop': 48, 'Putting a laptop somewhere': 49, 'Taking a laptop from somewhere': 50, 'Watching a laptop or something on a laptop': 51, 'Working/Playing on a laptop': 52, 'Holding a shoe/shoes': 53, 'Putting shoes somewhere': 54, 'Putting on shoe/shoes': 55, 'Taking shoes from somewhere': 56, 'Taking off some shoes': 57, 'Throwing shoes somewhere': 58, 'Sitting in a chair': 59, 'Standing on a chair': 60, 'Holding some food': 61, 'Putting some food somewhere': 62, 'Taking food from somewhere': 63, 'Throwing food somewhere': 64, 'Eating a sandwich': 65, 'Making a sandwich': 66, 'Holding a sandwich': 67, 'Putting a sandwich somewhere': 68, 'Taking a sandwich from somewhere': 69, 'Holding a blanket': 70, 'Putting a blanket somewhere': 71, 'Snuggling with a blanket': 72, 'Taking a blanket from somewhere': 73, 'Throwing a blanket somewhere': 74, 'Tidying up a blanket/s': 75, 'Holding a pillow': 76, 'Putting a pillow somewhere': 77, 'Snuggling with a pillow': 78, 'Taking a pillow from somewhere': 79, 'Throwing a pillow somewhere': 80, 'Putting something on a shelf': 81, 'Tidying a shelf or something on a shelf': 82, 'Reaching for and grabbing a picture': 83, 'Holding a picture': 84, 'Laughing at a picture': 85, 'Putting a picture somewhere': 86, 'Taking a picture of something': 87, 'Watching/looking at a picture': 88, 'Closing a window': 89, 'Opening a window': 90, 'Washing a window': 91, 'Watching/Looking outside of a window': 92, 'Holding a mirror': 93, 'Smiling in a mirror': 94, 'Washing a mirror': 95, 'Watching something/someone/themselves in a mirror': 96, 'Walking through a doorway': 97, 'Holding a broom': 98, 'Putting a broom somewhere': 99, 'Taking a broom from somewhere': 100, 'Throwing a broom somewhere': 101, 'Tidying up with a broom': 102, 'Fixing a light': 103, 'Turning on a light': 104, 'Turning off a light': 105, 'Drinking from a cup/glass/bottle': 106, 'Holding a cup/glass/bottle of something': 107, 'Pouring something into a cup/glass/bottle': 108, 'Putting a cup/glass/bottle somewhere': 109, 'Taking a cup/glass/bottle from somewhere': 110, 'Washing a cup/glass/bottle': 111, 'Closing a closet/cabinet': 112, 'Opening a closet/cabinet': 113, 'Tidying up a closet/cabinet': 114, 'Someone is holding a paper/notebook': 115, 'Putting their paper/notebook somewhere': 116, 'Taking paper/notebook from somewhere': 117, 'Holding a dish': 118, 'Putting a dish/es somewhere': 119, 'Taking a dish/es from somewhere': 120, 'Wash a dish/dishes': 121, 'Lying on a sofa/couch': 122, 'Sitting on sofa/couch': 123, 'Lying on the floor': 124, 'Sitting on the floor': 125, 'Throwing something on the floor': 126, 'Tidying something on the floor': 127, 'Holding some medicine': 128, 'Taking/consuming some medicine': 129, 'Putting groceries somewhere': 130, 'Laughing at television': 131, 'Watching television': 132, 'Someone is awakening in bed': 133, 'Lying on a bed': 134, 'Sitting in a bed': 135, 'Fixing a vacuum': 136, 'Holding a vacuum': 137, 'Taking a vacuum from somewhere': 138, 'Washing their hands': 139, 'Fixing a doorknob': 140, 'Grasping onto a doorknob': 141, 'Closing a refrigerator': 142, 'Opening a refrigerator': 143, 'Fixing their hair': 144, 'Working on paper/notebook': 145, 'Someone is awakening somewhere': 146, 'Someone is cooking something': 147, 'Someone is dressing': 148, 'Someone is laughing': 149, 'Someone is running somewhere': 150, 'Someone is going from standing to sitting': 151, 'Someone is smiling': 152, 'Someone is sneezing': 153, 'Someone is standing up from somewhere': 154, 'Someone is undressing': 155, 'Someone is eating something': 156}

charades_classes = list(charades_dict.keys())

split_75_train_list , split_25_test_list = np.split(charades_classes, [int(len(charades_classes)*0.75)])
split_50_train_list , split_50_test_list = np.split(charades_classes, [int(len(charades_classes)*0.50)])

charades_75_train_dict = { split_75_train_list[i] : i for i in sorted(range(len(split_75_train_list)))}
charades_25_test_dict = { split_25_test_list[i] : i for i in sorted(range(len(split_25_test_list)))}

charades_50_train_dict = { split_50_train_list[i] : i for i in sorted(range(len(split_50_train_list)))}
charades_50_test_dict = { split_50_test_list[i] : i for i in sorted(range(len(split_50_test_list)))}

