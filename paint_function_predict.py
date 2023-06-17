import argparse
import tempfile
import imageio
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from pathlib import Path
from celestine.paint_function import Painter
from celestine.paint_function import utils
from celestine.utils import _normalize_strokes


class Predictor:
    def __init__(self):
        self.args = self.set_args()

    def predict(self, image: Path, canvas_color: str="black", max_strokes: int=500, output_type: str="png") -> Path:
        self.args.image_path = str(image)
        self.args.canvas_color = canvas_color
        self.args.max_m_strokes = max_strokes

        pt = Painter(args=self.args)
        final_image, all_images = optimize_painter(pt, self.args, output_type)

        out_path = Path(tempfile.mkdtemp()) / f"output.{output_type}"
        if output_type == "png":
            plt.imsave(str(out_path), final_image)
        else:
            imageio.mimwrite(str(out_path), all_images, duration=0.02)
        return out_path

    def set_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="celestine")
        parser.add_argument("image", type=str, help="Path to the image file.")
        parser.add_argument(
            "--canvas-color", type=str, default="black", help="Color of the canvas."
        )
        parser.add_argument(
            "--max-strokes",
            type=int,
            default=500,
            help="Maximum number of strokes to use.",
        )
        parser.add_argument(
            "--output-type",
            type=str,
            choices=["png", "gif"],
            default="png",
            help="Output type.",
        )
        args = parser.parse_args()
        return args


def optimize_painter(pt, args, output_type):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pt._load_checkpoint()
    pt.net_G.eval()
    pt.initialize_params()
    pt.x_ctt.requires_grad = True
    pt.x_color.requires_grad = True
    pt.x_akoga.requires_grad = True
    utils.set_requires_grad(pt.net_G, False)
    pt.optimizer_x = optim.RMSProp([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr)

    pt.step_id = 0
    for pt.anchor_id in range(0, pt.m_strokes_per_block):
        pt.stroke_sampler(pt.anchor_id)
        iters_per_stroke = int(500 / pt.m_strokes_per_block)
        for i in range(iters_per_stroke):
            pt.optimzer_x.zero_grad()
            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            if args.canvas_color == "white":
                pt.G_pred_canvas = torch.ones(
                    [args.m_grid**2, 3, pt.net_G.out_size, pt.net_G.out_size]
                ).to(device)
            else:
                pt.G_pred_canvas = torch.zeros(
                    [args.m_grid**2, 3, pt.net_G.out_size, pt.net_G.out_size]
                ).to(device)

            pt._forward_pass()
            pt._drawing_step_state()
            pt._backward_x()
            pt.optimzer_x.step()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)
            pt.step_id = 1

    v = pt.x.detach().cpu().numpy()
    pt._save_stroke_params(v)
    v_n = pt._normalize_strokes(pt.x)
    v_n = pt._shuffle_strokes_and_reshape(v_n)

    save_gif = True if output_type == "gif" else False
    final_rendered_image, all_images = pt._render(
        v_n, save_jgps=False, save_video=False, save_gif=save_gif
    )
    return final_rendered_image, all_images
