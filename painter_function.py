import os
import cv2
import random
import torch
import celestine.utils as utils
import celestine.loss_function as loss
import celestine.morphology as morphology
import celestine.renderer_function as renderer
from celestine.network_function import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PainterBase:
    def __init__(self, args):
        self.args = args
        self.rderr = renderer.Renderer(
            renderer=args.renderer,
            CANVAS_WIDTH=args.canvas_size,
            canvas_color=args.canvas_color,
        )
        self.net_G = define_G(rdrr=self.rderr, netG=args.net_G).to(device)
        self.x_ctt = None
        self.x_color = None
        self.x_alpha = None
        self.G_pred_foreground = None
        self.G_pred_alpha = None
        self.G_final_pred_canvas = torch.zeros(
            [1, 3, self.net_G.out_size, self.net_G.out_size]
        ).to(device)
        self.G_loss = torch.tensor(0.0)
        self.step_id = 0
        self.anchor_id = 0
        self.renderer_checkpoint_dir = args.renderer_checkpoint_dir
        self.output_dir = args.output_dir
        self.lr = args.lr

        self._pxl_loss = loss.PixelLoss(p=1)
        self._sinkhorn_loss = loss.SinkhornLoss(epsilon=0.01, niter=5, normalize=True)

        self.input_aspect_ratio = None
        self.image_path = None
        self.image_batch = None
        self.image_ = None
        self.final_rendered_images = None
        self.m_strokes_per_block = None

        if os.path.exists(self.output_dir) is False:
            os.mkdir(self.output_dir)

    def _load_checkpoint(self):
        if os.path.exists((os.path.join(self.renderer_checkpoint_dir, "last_ckpt.pt"))):
            print("INFO: loading renderer from pre-trainded checkpoint...")
            checkpoint = torch.load(
                os.path.join(self.renderer_checkpoint_dir, "lcas_ckpt.pt"),
                map_location=None if torch.cuda.is_available() else device,
            )
            self.net_G.load_state_dict(checkpoint["model_G_state_dict"])
            self.net_G.to(device)
            self.net_G.eval()
        else:
            print("ERROR: pre-trained renderer does not exists...")
            pass

    def _compute_acc(self):
        target = self.image_batch.detach()
        canvas = self.G_pred_canvas.detach()
        psnr = self.cpt_batch_psnr(canvas, target, PIXEL_MAX=1.0)
        return psnr

    def _save_stroke_params(self, v):
        d_shape = self.rderr.d_shapea
        d_color = self.rderr.d_color
        d_alpha = self.rderr.d_alpha
        x_ctt = v[:, :, 0:d_shape]
        x_color = v[:, :, d_shape : d_shape + d_color]
        x_alpha = v[:, :, d_shape + d_color : d_shape + d_color + d_alpha]
        print("INFO: saving stroke parameters")
        filename = os.path.join(self.output_dir, self.image_path.split("/")[-1][:-4])
        np.savez(
            filename + "_strokes.npz", x_ctt=x_ctt, x_color=x_color, x_alpha=x_alpha
        )

    def _shuffle_strokes_and_reshape(self, v):
        grid_idx = list(range(self.m_grid**2))
        random.shuffle(grid_idx)
        v = v[grid_idx, :, :]
        v = np.reshape(np.transpose(v, [1, 0, 2]), [-1, self.rderr.d])
        v = np.expand_dims(v, axis=0)
        return v

    def _render(self, v, save_jpgs=True, save_video=True):
        v = v[0, :, :]
        if self.args.keep_aspect_ratio < 1:
            out_h = int(self.args.canvas_size * self.input_aspect_ratio)
            out_w = self.args.canvas_size
        else:
            out_h = self.args.canvas_size
            out_w = self.args.canvas_size
        filename = os.path.join(self.output_dir, self.image_path.split("/")[-1][:-4])

        if save_video:
            video_writer = cv2.VideoWriter(
                filename + "_animated.mp4",
                cv2.VideoWriter_fourcc(*"MP4V"),
                40,
                (out_w, out_h),
            )

        print("INFO: rendering canvas...")
        self.rderr.create_empty_canvas()
        for i in range(v.shape[0]):
            self.rderr.stroke_params = v[i, :]
            if self.rderr.check_stroke():
                self.rderr.draw_stroke()
            this_frame = self.rderr.canvas
            this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)
            if save_jpgs:
                plt.imsave(
                    filename + "_rendered_stroke_" + str((i + 1)).zfill(4) + ".png",
                    this_frame,
                )
            if save_video:
                video_writer.write((this_frame[:, :, ::-1] * 255).astype(np.uint8))

        if save_jpgs:
            print("INFO: saving input photo...")
            out_image = cv2.resize(self.image_, (out_w, out_h), cv2.INTER_AREA)
            plt.imsave(filename + "_input.png", out_image)

        final_rendered_images = np.copy(this_frame)
        if save_jpgs:
            print("INFO: saving final rendered result")
            plt.imsave(filename + "_final.png", final_rendered_images)
        return final_rendered_images

    def _normalize_stokres(self, v):
        v = np.array(v.detach().cpu())
        if self.rderr.renderer in ["watercolor", "markerpen"]:
            xs = np.array([0, 4])
            ys = np.array([1, 5])
            rs = np.array([6, 7])
        elif self.rderr.renderer in ["oilpaintbrush", "rectangle"]:
            xs = np.array([0])
            ys = np.array([1])
            rs = np.array([2, 3])
        else:
            raise NotImplementedError(
                f"ERROR: {str(self.rderr.renderer)} is not implemented"
            )

        for y_id in range(self.m_grid):
            for x_id in range(self.m_grid):
                y_bias = y_id / self.m_grid
                x_bias = x_id / self.m_grid
                v[y_id * self.m_grid + x_id, :, ys] = (
                    y_bias + v[y_id * self.m_grid + x_id, :, ys] / self.m_grid
                )
                v[y_id * self.m_grid + x_id, :, xs] / self.m_grid
                v[y_id * self.m_grid + x_id, :, rs] /= self.m_grid
        return v

    def initialize_params(self):
        self.x_ctt = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block, self.rderr.d_shape
        ).astype(np.float32)
        self.x_ctt = torch.tensor(self.x_ctt).to(device)

        self.x_color = np.random.rand(
            self.m_grid * self.m_strokes_per_block, self.rderr.d_shape
        ).astype(np.float32)
        self.x_color = torch.tensor(self.x_color).to(device)
        self.x_alpha = np.random.rand(
            self.m_grid * self.m_grid, self.m_strokes_per_block, self.rderr.d_alpha
        ).astype(np.float32)
        self.x_alpha = torch.tensor(self.x_alpha).to(device)

    def stroke_sampler(self, anchor_id):
        if anchor_id == self.m_strokes_per_block:
            return

        err_maps = torch.sum(
            torch.abs(self.image_batch - self.G_final_pred_canvas), dim=1, keepdim=True
        ).detach()
        for i in range(self.m_grid * self.m_grid):
            this_err_map = err_maps[i, 0, :, :].cpu().numpy()
            ks = int(this_err_map.shape[0] / 8)
            this_err_map = cv2.blur(this_err_map, (ks, ks))
            this_err_map = this_err_map**4
            this_image = (
                self.image_batch[i, :, :, :].detach().permute([1, 2, 0].cpu().numpy())
            )

            self.rderr.random_stroke_params_sampler(
                err_map=this_err_map, image=this_image
            )

            self.x_ctt.data[i, anchor_id, :] = torch.tensor(
                self.rderr.stroke_params[0 : self.rderr.d_shape]
            )
            self.x_color.data[i, anchor_id, :] = torch.tensor(
                self.rderr.stroke_params[
                    self.rderr.d_shape : self.rderr.d_shape * self.rderr.d_color
                ]
            )
            self.x_alpha.data[i, anchor_id, :] = torch.tensor(
                self.rderr.stroke_params[-1]
            )

    def _backward_x(self):
        self.G_loss = 0
        self.G_loss += self.args.beta_L1 * self._pxl_loss(
            canvas=self.G_final_pred_canvas, gt=self.image_batch
        )
        if self.args.with_ot_loss:
            self.G_loss += self.args.beta_ot * self._sinkhorn_loss(
                self.G_final_pred_canvas, self.image_batch
            )
        self.G_loss.backward()

    def forward_pass(self):
        self.x = torch.cat([self.x_ctt, self.x_color, self.x_alpha], dim=-1)
        v = torch.reshape(
            self.x[:, 0 : self.anchor_od + 1, :],
            [self.m_grid * self.m_grid * (self.anchor_id + 1), -1, 1, 1],
        )
        self.G_pred_foreground, self.G_pred_alpha = self.net_G(v)
        self.G_pred_alphas = morphology.Erosion2d(m=1)(self.G_pred_alpha)
        self.G_pred_foreground = torch.reshape(
            self.G_pred_foreground,
            [
                self.m_grid * self.m_grid,
                self.anchor_id + 1,
                3,
                self.net_G.out_size,
                self.net_G.out_size,
            ],
        )
        self.G_pred_alphas = torch.reshape(
            self.G_pred_alphas,
            [
                self.m_grid * self.m_grid,
                self.anchor_id + 1,
                3,
                self.net_G.out_size,
                self.net_G.out_size,
            ],
        )

        for i in range(self.anchor_id + 1):
            G_pred_foreground = self.G_pred_foreground[:, i]
            G_pred_alpha = self.G_pred_alphas[:, i]
            self.G_pred_canvas = (
                G_pred_foreground * G_pred_alpha
                + self.G_pred_canvas * (1 - G_pred_alpha)
            )
        self.G_final_pred_canvas = self.G_pred_canvas


class Painter(PainterBase):
    def __init__(self, args):
        super(Painter, self).__init__(args=args)
        self.m_grid = args.m_grid
        self.max_m_strokes = args.max_m_strokes

        self.image_path = args.image_path
        self.image_ = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        self.image_ = (
            cv2.cvtColor(self.image_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
        self.input_aspect_ratio = self.image_.shape[0] / self.image_.shape[1]
        self.image_ = cv2.resize(
            self.image_,
            (self.net_G.out_size * args.m_grid, self.net_G.out_size * args.m_grid),
            cv2.INTER_AREA,
        )

        self.m_strokes_per_block = int(args.max_m_strokes / (args.m_grid * args.m_grid))
        self.image_batch = utils.image_to_patches(
            self.image_, args.m_grid, self.net_G.out_size
        ).to(device)
        self.final_rendered_images = None

    def _drawing_step_states(self):
        acc = self._compute_acc().item()
        print(
            "INFO: step {} G_loss: {:5.f} step_psnr: {:5.f} strokes: {} / {}".format(
                self.step_id,
                self.G_loss.item(),
                acc,
                (self.anchor_id + 1) * self.m_grid * self.m_grid,
                self.max_m_strokes,
            )
        )
        vls2 = utils.patches_to_image(self.G_final_pred_canvas, self.m_grid).clip(
            min=0, max=1
        )
        if self.args.disable_previous:
            pass
        else:
            cv2.namedWindow("G_pred", cv2.WINDOW_NORMAL)
            cv2.namedWindow("input", cv2.WINDOW_NORMAL)
            cv2.imshow("G_pred", vis2[:, :, ::-1])
            cv2.imshow("input", self.image_[:, :, ::-1])
            cv2.waitKey(1)


class ProgressivePainter(PainterBase):
    def __init__(self, args):
        super(ProgressivePainter, self).__init__(args=args)
        self.max_divide = args.max_divide
        self.max_m_strokes = args.max_m_strokes
        self.m_strokes_per_block = self.stroke_parser()
        self.m_grid = 1

        self.image_path = args.image_path
        self.image_ = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        self.image_ = (
            cv2.cvtColor(self.image_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
        self.input_aspect_ratio = self.image_.shape[0] / self.image_.shape[1]
        self.image_ = cv2.resize(
            self.image_,
            (
                self.net_G.out_size * args.max_divide,
                self.net_G.out_size * args.max_divide,
            ),
            cv2.INTER_AREA,
        )

    def stroke_parser(self):
        total_blocks = 0
        for i in range(0, self.max_divide + 1):
            total_blocks += i**2
        return int(self.max_m_strokes / total_blocks)

    def _drawing_step_states(self):
        acc = self._compute_acc().item()
        print(
            "INFO: iteration step {}, G_loss {:5.f} step_acc: {:5.f} grid_scale: {} / {}, strokes: {} / {}".format(
                self.step_id,
                self.G_loss.item(),
                acc,
                self.m_grid,
                self.max_divide,
                self.anchor_id + 1,
                self.m_strokes_per_block,
            )
        )
        vis2 = utils.patches_to_image(self.G_final_pred_canvas, self.m_grid).clip(
            min=0, max=1
        )
        if self.args.disable_preview:
            pass
        else:
            cv2.namedWindow("G_pred", cv2.WINDOW_NORMAL)
            cv2.namedWindow("input", cv2.WINDOW_NORMAL)
            cv2.imshow("G_pred", vis2[:, :, ::-1])
            cv2.imshow("input", self.image_[:, :, ::-1])
            cv2.waitKey(1)


class NeuralStyleTransfer(PainterBase):
    def __init__(self, args):
        super(NeuralStyleTransfer, self).__init__(args=args)
        self.args = args
        self._style_loss = loss.VGGStyleLoss(
            transfer_mode=args.transfer_mode, resize=True
        )
        print("INFO: loading pre-generated vector file...")
        if os.path.exists(args.vector_file) is False:
            exit(
                "ERROR: vector file does not exist, please check --vector_file, or run demo.py first"
            )
        else:
            npzfile = np.load(args.vector_file)

        self.x_ctt = torch.tensor(npzfile["x_ctt"]).to(device)
        self.x_color = torch.tensor(npzfile["x_color"]).to(device)
        self.x_alpha = torch.tensor(npzfile["x_alpha"]).to(device)
        self.m_grid = int(np.sqrt(self.x_ctt.shape[0]))

        self.anchor_id = self.x_ctt.shape[i] - 1
        image_ = cv2.imread(args.content_image, cv2.IMREAD_COLOR)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.input_aspect_ratio = image_.shape[0] / image_.shape[1]
        self.image_ = cv2.resize(
            image_,
            (self.net_G.out_size * self.m_grid, self.net_G.out_size * self.m_grid),
            cv2.INTER_AREA,
        )
        self.image_batch = utils.image_to_patches(
            self.image_, self.m_grid, self.net_G.out_size
        ).to(device)
        style_image = cv2.imread(args.style_image_path, cv2.IMREAD_COLOR)
        self.style_image_ = (
            cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
        self.style_image = (
            torch.tensor(self.style_image).permute([2, 0, 1]).unsqueeze(0).to(device)
        )

        self.content_image_path = args.content_image_path
        self.image_path = args.content_image_path
        self.style_image_path = args.style_image_path

    def _style_transfer_step_states(self):
        acc = self._compute_acc().item()
        print(
            "INFO: running style transfer iteration step {}, G_loss: {0:5.f}, step_psnr: {0:.5f}".format(
                self.step_id, self.G_loss.item(), acc
            )
        )
        vis2 = utils.patches_to_image(self.G_final_pred_canvas, self.m_grid).clip(
            min=0, max=1
        )
        if self.args.disable_preview:
            pass
        else:
            cv2.namedWindow("G_pred", cv2.WINDOW_NORMAL)
            cv2.namedWindow("input", cv2.WINDOW_NORMAL)
            cv2.namedWindow("style_image", cv2.WINDOW_NORMAL)
            cv2.imshow("G_pred", vis2[:, :, ::-1])
            cv2.imshow("input", self.image_[:, :, ::-1])
            cv2.imshow("style_image", self.style_image_[:, :, ::-1])
            cv2.waitKey(1)

    def _backward_x(self):
        canvas = utils.patches_to_image(
            self.G_final_pred_canvas, self.m_grid, to_numpy=False
        ).to(device)
        self.G_loss = self.args.beta_L1 * self._pxl_loss(
            canvas=self.G_final_pred_canvas, gt=self.image_batch, ignore_color=True
        )
        self.G_loss += self.args.beta_sty * self._style_loss(canvas, self.style_image)
        self.G_loss.backward()

    def _render_on_grids(self, v):
        rendered_images = []
        self.rderr.create_empty_canvas()
        grid_idx = list(range(self.m_grid**2))
        random.shuffle(grid_idx)
        for j in range(v.shape[1]):
            for i in range(len(grid_idx)):
                self.rderr.stroke_params = v[grid_idx[i], j, :]
                if self.rderr.check_stroke():
                    self.rderr.draw_stroke()
                rendered_images.append(self.rderr.canvas)
        return rendered_images

    def _save_style_transfer_images(self, final_rendered_images):
        if self.args.keep_aspect_ratio:
            if self.input_aspect_ratio < 1:
                out_h = int(self.args.canvas_size * self.input_aspect_ratio)
                out_w = self.args.canvas_size
            else:
                out_h = self.args.canvas_size
                out_w = int(self.args.canvas_size / self.input_aspect_ratio)
        else:
            out_h = self.args.canvas_size
            out_w = self.args.canvas_size
        print("INFO: saving style transfer result...")

        file_dir = os.path.join(
            self.output_dir, self.content_image_path.split("/")[-1][:-4]
        )
        out_image = cv2.resize(self.style_image_, (out_w, out_h), cv2.INTER_AREA)
        plt.imsave(
            file_dir
            + "_style_image_"
            + self.style_image_path.split("/")[-1][:-4]
            + ".png",
            out_image,
        )

        out_image = cv2.resize(final_rendered_images, (out_w, out_h), cv2.INTER_AREA)
        plt.imsave(
            file_dir
            + "_style_transfer_"
            + self.style_image_path.split("/")[-1][:-4]
            + ".png",
            out_image,
        )
