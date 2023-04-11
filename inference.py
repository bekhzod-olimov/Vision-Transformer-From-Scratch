# Import libraries
from PIL import Image
import torch, cv2
from torchvision import transforms as T

# Set k to compute topk accuracy score
k = 10

# Get class names
classes = dict(enumerate(open("imagenet_classes.txt")))

# Load a trained model
model = torch.load("test.pth")

# Switch the model into evaluation mode
model.eval()

# Read an image
im = Image.open("bear.jpg")

# Initialize transformations
tfs = T.Compose([T.Resize((384, 384)), T.ToTensor()])

# Apply transformations and input to the model
out = torch.nn.functional.softmax(model(tfs(im).unsqueeze(0)), dim = -1)

# Get topk values and indices
vals, ids = out[0].topk(k)

# Go through topk values and indices
for idx, (val, id) in enumerate(zip(vals, ids)):
    lbl = classes[id.item()].strip()
    print(f"Top {idx+1} -> {lbl} with a score of {val.item():.3f}")
