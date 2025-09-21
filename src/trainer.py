import torch

class Trainer:
    def __init__(self, model, device, optimizer, criterion_box, criterion_cls, alpha=0.5, beta=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion_box = criterion_box
        self.criterion_cls = criterion_cls

        self.alpha = alpha
        self.beta = beta if beta is not None else (1 - alpha)

    def init_train_parameters(self):
        self.train_loss = 0
        self.train_loss_box = 0
        self.train_loss_cls = 0
        self.train_accuracy = 0
        self.train_n_sample = 0

        self.eval_loss = 0
        self.eval_loss_box = 0
        self.eval_loss_cls = 0
        self.eval_accuracy = 0
        self.eval_n_sample = 0

    def train_step(self, batch):
        x, y_cls, y_box = batch
        x, y_cls, y_box = x.to(self.device), y_cls.to(self.device), y_box.to(self.device)
        model = self.model.to(self.device)

        model.train()
        self.optimizer.zero_grad()
        pred_cls, pred_box = model(x)
        box_loss = self.criterion_box(pred_box, y_box)
        cls_loss = self.criterion_cls(pred_cls, y_cls.squeeze(-1))

        loss = self.alpha * box_loss + self.beta * cls_loss
        loss.backward()

        self.optimizer.step()
        self.train_loss += loss.item()
        self.train_loss_box += box_loss.item()
        self.train_loss_cls += cls_loss.item()
        self.train_accuracy += y_cls.eq(pred_cls.argmax(dim=-1, keepdims=True)).sum().item()
        self.train_n_sample += len(y_cls)

    def eval_step(self, batch):
        x, y_cls, y_box = batch
        x, y_cls, y_box = x.to(self.device), y_cls.to(self.device), y_box.to(self.device)
        model = self.model.to(self.device)

        model.eval()
        with torch.no_grad():
            pred_cls, pred_box = model(x)
        box_loss = self.criterion_box(pred_box, y_box)
        cls_loss = self.criterion_cls(pred_cls, y_cls.squeeze(-1))
        loss = self.alpha * box_loss + self.beta * cls_loss

        self.eval_loss += loss.item()
        self.eval_loss_box += box_loss.item()
        self.eval_loss_cls += cls_loss.item()
        self.eval_accuracy += y_cls.eq(pred_cls.argmax(dim=-1, keepdims=True)).sum().item()
        self.eval_n_sample += len(y_cls)

