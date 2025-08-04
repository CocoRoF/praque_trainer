from trl import SFTTrainer
import torch
import copy
from torch.nn.parallel import DistributedDataParallel as DDP

class KL_SFTTrainer(SFTTrainer):
    def __init__(self, *args, kl_alpha=0.2, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self.model, DDP):
            print("[INFO] Using DistributedDataParallel Model. Copying the module.")
            student_model_to_copy = self.model.module
        else:
            print("[INFO] Using Single Model. No need to copy.")
            student_model_to_copy = self.model

        self.teacher_model = copy.deepcopy(student_model_to_copy)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.kl_alpha = kl_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        with torch.no_grad():
            print("[INFO] Using Teacher Model for KL Divergence Loss Calculation")
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs["logits"]

        student_logits = outputs["logits"]

        T = 1.0
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits / T, dim=-1),
            torch.nn.functional.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean"
        ) * (T ** 2)

        total_loss = (loss * (1-self.kl_alpha)) + (self.kl_alpha * kl_loss)
        return (total_loss, outputs) if return_outputs else total_loss
