import torch
import torch.nn.functional as F

def predict_frame(model, face, device):
    face = torch.tensor(face).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(face)
        prob = F.softmax(out, dim=1)

    return prob.argmax(1).item(), prob[0][1].item()
