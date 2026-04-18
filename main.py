import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Upload images
uploaded = files.upload()
names = list(uploaded.keys())

content_name = [n for n in names if "content" in n.lower()][0]
style_name = [n for n in names if "style" in n.lower()][0]

# Load image
def load_img(path):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

content = load_img(content_name)
style = load_img(style_name)

# Model
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

def get_features(x):
    layers = ['0','5','10','19','21']
    features = []
    for i, layer in model._modules.items():
        x = layer(x)
        if i in layers:
            features.append(x)
    return features

def gram(x):
    b,c,h,w = x.size()
    x = x.view(c, h*w)
    return torch.mm(x, x.t())

content_f = get_features(content)
style_f = get_features(style)

target = content.clone().requires_grad_(True)
optimizer = torch.optim.Adam([target], lr=0.003)

for i in range(80):  # reduced steps (faster)
    target_f = get_features(target)
    
    content_loss = torch.mean((target_f[-1] - content_f[-1])**2)
    
    style_loss = 0
    for t,s in zip(target_f, style_f):
        style_loss += torch.mean((gram(t) - gram(s))**2)
    
    loss = content_loss + 1e5 * style_loss
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if i % 20 == 0:
        print("Step:", i, "Loss:", loss.item())

img = target.squeeze().detach().cpu()
img = transforms.ToPILImage()(img)
img.save("output.jpg")

plt.imshow(img)
plt.axis('off')
plt.show()

files.download("output.jpg")