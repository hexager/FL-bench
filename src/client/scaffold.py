from typing import Any

import torch

from src.client.fedavg import FedAvgClient


class SCAFFOLDClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.c_local: list[torch.Tensor]
        self.c_global: list[torch.Tensor]
        self.y_delta: list[torch.Tensor]
        self.c_delta: list[torch.Tensor]
        self.num_local_steps = 0
        self.scaffold_lr = self.args.optimizer.lr

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.c_global = package["c_global"]
        self.c_local = package["c_local"]
        # Reset per-call accounting even when local_epoch == 0 and fit() is skipped.
        self.num_local_steps = 0

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()

        with torch.no_grad():
            self.y_delta = []
            self.c_delta = []

            model_params = self.model.state_dict()
            for key in server_package["regular_model_params"].keys():
                global_param = server_package["regular_model_params"][key]
                local_param = model_params[key]
                self.y_delta.append(local_param.cpu() - global_param)

            # If the client performed no optimization steps (e.g. a 0-epoch
            # straggler), it must not change either its model/control update.
            if self.num_local_steps == 0:
                self.c_delta = [torch.zeros_like(c_i) for c_i in self.c_local]
            else:
                # SCAFFOLD Option II (Karimireddy et al., 2020):
                # c_i^+ = c_i - c + (x - y_i) / (K * eta_l)
                # y_delta stores (y_i - x), hence the minus sign below.
                coef = 1.0 / (self.num_local_steps * self.scaffold_lr)
                c_plus = [
                    c_i - c - coef * y_del
                    for c, c_i, y_del in zip(
                        self.c_global, self.c_local, self.y_delta
                    )
                ]
                self.c_delta = [
                    c_p - c_i for c_p, c_i in zip(c_plus, self.c_local)
                ]
                self.c_local = c_plus

        return self.package()

    def package(self):
        client_package = super().package()
        client_package["c_delta"] = [c.clone().cpu() for c in self.c_delta]
        client_package["y_delta"] = [y.clone().cpu() for y in self.y_delta]
        # The server owns persistent per-client state. Return the updated c_i
        # explicitly so it is not lost when workers/clients are reused.
        client_package["c_local"] = [c.clone().cpu() for c in self.c_local]
        client_package["num_local_steps"] = self.num_local_steps
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        self.num_local_steps = 0

        # SCAFFOLD's control-variate update assumes a fixed local learning rate.
        # FL-bench's default scheduler is disabled; record the actual optimizer
        # LR used for this local training call rather than relying on config only.
        self.scaffold_lr = self.optimizer.param_groups[0]["lr"]

        # Match FL-bench's common.local_epoch semantics used by FedAvg/FedProx:
        # each local epoch is a complete pass through the client's trainloader.
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # BatchNorm2d cannot train on a singleton batch.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()

                for param, c, c_i in zip(
                    self.model.parameters(), self.c_global, self.c_local
                ):
                    if param.requires_grad:
                        param.grad.data += (c - c_i).to(self.device)

                self.optimizer.step()
                self.num_local_steps += 1

            # Keep scheduler behavior consistent with FedAvg/FedProx: once per
            # completed local epoch, not once per minibatch.
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
