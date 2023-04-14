import torch


class Trainer(object):
    def __init__(self, epoch=1, optimizer=None, loss=None, metrics=None, device="cpu"):
        self.epoch = epoch
        self.optimizer = optimizer
        self.criterion = loss
        self.device = device
        self.metrics = metrics
        self.model = None
        self.decoder = None
        self.dataloader = None
        self.convert = None

    def _eval(self, *args, **kwargs):
        self.model.eval()
        self.decoder.eval()
        with torch.no_grad():
            for batch_nb, batch_data in enumerate(self.dataloader):
                self._step(batch_data, *args, **kwargs)

    def _test(self, *args, **kwargs):
        self.model.eval()
        self.decoder.eval()
        with torch.no_grad():
            for batch_nb, batch_data in enumerate(self.dataloader):
                self._step(batch_data, *args, **kwargs)

    def _step(self, batch_data, *args, **kwargs):
        ori_str, feature, label = self._pre_process_data(batch_data, *args, **kwargs)
        # model = model if model else self.model
        with torch.no_grad():
            emb = self.model(feature)
        pred = self.decoder(emb, ori_str)
        loss = self.criterion(pred, label)
        return pred, label, loss

    def _pre_process_data(self, batch_data, *args, **kwargs):
        ori_str, label, *_ = batch_data
        label = label.float()
        feature = kwargs["convert"](ori_str) if "convert" in kwargs else ori_str
        # feature = feature.int()
        feature = feature.to(self.device)
        label = label.to(self.device)
        return ori_str, feature, label

    def predict(self, dataloader, pre_model=None, down_model=None, *args, **kwargs):
        dl = dataloader if dataloader else self.dataloader
        self.model = pre_model.to(self.device) if pre_model else self.model.to(self.device)
        self.model.eval()
        self.decoder = down_model.to(self.device) if down_model else self.decoder.to(self.device)
        self.decoder.eval()
        pred, label = [], []
        with torch.no_grad():
            for batch_nb, batch_data in enumerate(dl):
                # TODO: 考虑数据需不需要预处理
                pred, label, *_ = self._step(batch_data, *args, **kwargs)
                pred.extend(pred.view(1, -1).tolist())
                label.extend(label.view(-1).tolist())
        return {"pred": pred, "label": label}

    def score(self, dataloader, pre_model=None, down_model=None, *args, **kwargs):
        result = self.predict(dataloader, pre_model, down_model, *args, **kwargs)
        return self.metrics(result["pred"], result["label"])

