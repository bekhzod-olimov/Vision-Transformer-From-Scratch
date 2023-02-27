from PIL import Image
import torch, cv2
from torchvision import transforms as T

k = 10

classes = dict(enumerate(open("imagenet_classes.txt")))
model = torch.load("test.pth")
model.eval()

im = Image.open("bear.jpg")
tfs = T.Compose([T.Resize((384, 384)), T.ToTensor()])
inp = tfs(im)
out = torch.nn.functional.softmax(model(inp.unsqueeze(0)), dim = -1)

vals, ids = out[0].topk(k)

for idx, (val, id) in enumerate(zip(vals, ids)):
    
    lbl = classes[id.item()].strip()
    print(f"Top {idx+1} -> {lbl} with a score of {val.item():.3f}")
