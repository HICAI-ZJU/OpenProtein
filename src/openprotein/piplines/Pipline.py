from openprotein.piplines import Trainer


class Pipline(object):

    def __init__(self, pretrain, downstream, mode=None, device="cpu", **kwargs):
        self.pretrain = pretrain
        self.downstream = downstream
        self.mode = mode
        self.device = device
        self.trainer = Trainer(device=device)


    @classmethod
    def load_predict(cls, pretrain, downstream, checkpoints_dir=None, device="cpu"):
        if checkpoints_dir:
            pretrain.load_state_dict(torch.load(checkpoints_dir))
        mode = "predict"
        return cls(pretrain, downstream, mode, device)


    def _check_running_mode(self, running_mode):
        if self.mode != running_mode:
            raise RuntimeError(f"the mode is not {running_mode}, current mode is {self.mode}")


    def predict(self, dataloader, *args, **kwargs):
        # 预测下游任务数据
        """
        需要预测什么

        :param data:
        :return:
        """
        self._check_running_mode("predict")

        for name, parameter in self.pretrain.named_parameters():
            parameter.requires_grad = False
        for name, parameter in self.downstream.named_parameters():
            parameter.requires_grad = False

        result = self.trainer.predict(dataloader, self.pretrain, self.downstream, *args, **kwargs)
        return result
