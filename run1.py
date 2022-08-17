
from detecto import core, utils , visualize



model = core.Model.load('C:\\Users\\Hp\\Desktop\\Git Repos\\detecto_inno\\foto\\14_08_1_withweights.pth', ['hasar','korozyon'])


i = "C:\\Users\\Hp\\Desktop\\Git Repos\\detecto_inno\\foto\\4268019_1658392371557_20220721113252394_26.4339_38.6714.jpg"

visualize.file_path(f"{i}")

image =utils.read_image(f"{i}")
predictions = model.predict_top(image)

# predictions format: (labels, boxes, scores)
labels, boxes, scores = predictions

#label_list.append(labels)
#score_list.append(scores)
 
#print(labels)
#print(scores)

visualize.show_labeled_image(image, boxes[0:3], labels[0:3])