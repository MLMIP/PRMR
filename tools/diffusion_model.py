import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ddpm.trainer_unc import _build_model


DEFAULT_DIFFUSION_SIZE = (512, 256)
DEFAULT_OUTPUT_SIZE = (1080, 1920)


class DiffusionModel:
    def __init__(
        self,
        params_path,
        checkpoint_path,
        device="cuda",
        num_classes=20,
        small_class_threshold=0.01,
    ):
        self.device = device
        self.num_classes = num_classes
        self.small_class_threshold = small_class_threshold

        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

        self.model = _build_model(
            params=params,
            input_shapes=[(3, 256, 512), (num_classes, 256, 512)],
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.unet.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

    def calculate_confidence_mask(self, seg_logits, uncertainty, threshold):
        sorted_values = np.sort(seg_logits, axis=0)
        top2_values = sorted_values[-2:]

        top1, top2 = top2_values[1], top2_values[0]
        max_logits = top1 / (top1 + top2)

        second_max_indices = np.argpartition(seg_logits, -2, axis=0)[-2]
        mask_un = uncertainty > threshold

        return max_logits, mask_un, second_max_indices

    def identify_small_classes(self, result):
        mask_little = np.zeros_like(result, dtype=bool)

        for class_id in range(self.num_classes - 1):
            class_mask = result == class_id
            class_percent = np.sum(class_mask) / result.size

            if class_percent < self.small_class_threshold:
                mask_little[class_mask] = True

        return mask_little

    def swap_labels_by_mask(
        self,
        pseudo_label1,
        pseudo_label2,
        mask,
        confidence_logits,
    ):
        swap_probability = 1 - confidence_logits

        swapped_label1 = pseudo_label1.copy()
        swapped_label2 = pseudo_label2.copy()

        candidate_positions = np.where(mask)
        if len(candidate_positions[0]) == 0:
            return swapped_label1

        candidate_logits = swap_probability[candidate_positions]
        random_values = np.random.rand(len(candidate_logits))
        swap_flags = random_values < candidate_logits

        swap_positions_selected = tuple(pos[swap_flags] for pos in candidate_positions)

        if len(swap_positions_selected[0]) > 0:
            temp = swapped_label1[swap_positions_selected]
            swapped_label1[swap_positions_selected] = swapped_label2[swap_positions_selected]
            swapped_label2[swap_positions_selected] = temp

        return swapped_label1

    def preprocess_for_diffusion(self, result, target_size=DEFAULT_DIFFUSION_SIZE):
        result_resized = cv2.resize(
            result, target_size[::-1], interpolation=cv2.INTER_NEAREST
        )

        result_tensor = torch.tensor(result_resized).unsqueeze(0).long()
        x0 = torch.nn.functional.one_hot(result_tensor, self.num_classes)
        x0 = x0.permute(0, 3, 1, 2).float().to(self.device)

        return x0

    @torch.no_grad()
    def apply_diffusion(self, x0, time_step):
        t = torch.tensor([time_step], device=self.device)

        xt = self.model.diffusion.q_xt_given_x0(x0, t).sample()
        sample = self.model(xt.contiguous(), t)

        sample = sample["diffusion_out"].float()

        result = F.interpolate(
            sample,
            size=DEFAULT_OUTPUT_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        result = torch.argmax(result, dim=1)[0].cpu().numpy()

        return result

    def post_process(
        self,
        result,
        result_adapt,
        mask_un,
        second_max_indices,
        mask_little,
    ):
        result_final = result_adapt.copy()

        # keep confident regions
        result_final[~mask_un] = result[~mask_un]

        # remove inconsistent corrections
        mask_same = result_final == result
        mask_error = np.zeros_like(result_final, dtype=bool)

        different_positions = ~mask_same
        if np.any(different_positions):
            mask_error[different_positions] = (
                result_final[different_positions]
                != second_max_indices[different_positions]
            )
            result_final[mask_error] = result[mask_error]

        # protect small objects
        result_final[mask_little] = result[mask_little]
        return result_final

    def dpr(
        self,
        seg_logits,
        seg_logits_anchor,
        uncertainty,
        result,
        time_step=180,
        threshold=0.9,
    ):
        max_logits, mask_un, second_max_indices = self.calculate_confidence_mask(
            seg_logits, uncertainty, threshold
        )

        mask_little = self.identify_small_classes(result)

        result_copy = self.swap_labels_by_mask(
            result,
            second_max_indices,
            mask_un,
            max_logits,
        )

        result_copy[mask_little] = result[mask_little]
        x0 = self.preprocess_for_diffusion(result_copy)

        result_adapt = self.apply_diffusion(x0, time_step)

        result_final = self.post_process(
            result,
            result_adapt,
            mask_un,
            second_max_indices,
            mask_little,
        )

        return [result_final.astype(np.int64)]

