import time
import timm
import torch
from torch import nn
from torch.optim import Adam
from utils.utils import AverageMeter
from transformers import TimesformerModel, logging
from torch.nn.parallel import DistributedDataParallel as DDP


class TimeSformer(nn.Module):
    def __init__(self, num_frames, num_classes=10):
        super().__init__()
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", num_frames=num_frames, ignore_mismatched_sizes=True)
        self.classif = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=num_classes)

    def forward(self, images):
        x = self.backbone(images)[0]
        x = self.classif(x[:, 0])
        return x
    

class TimeSformerExecutor:
    def __init__(self, train_loader, test_loader, criterion, eval_metric, class_list, test_every, distributed, gpu_id) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metric = eval_metric.to(gpu_id)
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = gpu_id
        num_frames = self.train_loader.dataset[0][0].shape[0]
        num_classes = len(class_list)
        logging.set_verbosity_error()
        model = TimeSformer(num_frames=num_frames, num_classes=num_classes).to(gpu_id)
        if distributed: 
            self.model = DDP(model, device_ids=[gpu_id])
        else: 
            self.model = model
        for p in self.model.parameters():
            p.requires_grad = True
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": 0.00001}])

    def _train_batch(self, data, label):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss_this = self.criterion(output, label)
        loss_this.backward()
        self.optimizer.step()
        return loss_this.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        for data, label in self.train_loader:
            data, label = data.to(self.gpu_id), label.to(self.gpu_id)
            loss_this = self._train_batch(data, label)
            loss_meter.update(loss_this, data.shape[0])
        elapsed_time = time.time() - start_time
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            print("Epoch [" + str(epoch + 1) + "]"
                  + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                  + " loss: " + "{:.4f}".format(loss_meter.avg))
    
    def train(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval = self.test()
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100))
    
    def test(self):
        self.model.eval()
        eval_meter = AverageMeter()
        for data, label in self.test_loader:
            data, label = data.to(self.gpu_id), label.long().to(self.gpu_id)
            with torch.no_grad():
                output = self.model(data)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg
    
    def save(self, file_path="./checkpoint.pth"):
        backbone_state_dict = self.model.backbone.state_dict()
        classifier_state_dict = self.model.classifier.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": backbone_state_dict,
                    "classifier": classifier_state_dict,
                    "optimizer": optimizer_state_dict},
                    file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.backbone.load_state_dict(checkpoint["backbone"])
        self.model.transformer.load_state_dict(checkpoint["classifier"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])