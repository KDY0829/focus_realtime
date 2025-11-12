# src/emotion.py
import cv2, torch, timm
import numpy as np
from torchvision import transforms
from collections import OrderedDict

DEFAULT_LABELS = ["anger","disgust","fear","happy","none","sad","surprise"]

class EmotionPredictor:
    def __init__(self, weight_path, labels=DEFAULT_LABELS, img_size=384, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = labels
        self.model = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(labels))
        sd = torch.load(weight_path, map_location=self.device)
        # best가 EMA state_dict인 경우도 OK
        if isinstance(sd, OrderedDict):
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model.load_state_dict(sd.get("model", sd), strict=False)
        self.model.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # 정책: 부정(anger, disgust, fear, sad) 합을 감점 → 집중 점수
        self.negative = set(["anger","disgust","fear","sad"])

    def predict(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probs = {lab: float(p) for lab, p in zip(self.labels, prob)}
        # 집중 점수: 1 - sum(부정)
        neg_sum = sum(probs.get(k,0.0) for k in self.negative)
        score = float(np.clip(1.0 - neg_sum, 0.0, 1.0))
        top = max(probs, key=probs.get)
        return score, probs, top

