"""
painter package
"""
import os
import tkinter as tk
from tkinter import font
from .utils import parse_microbatch_key, save_to_file
from .abstract.mutils import *
from .PainterColor import set_color
import tkinter as tk
from PIL import Image, EpsImagePlugin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

import os
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
EpsImagePlugin.gs_windows_binary = "/opt/homebrew/bin/gs"  # 请根据你的机器修改路径

def save_canvas_as_pdf(canvas, filename="output.pdf", scale=5):
    canvas.update()
    
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    
    ps_filename = "temp_output.ps"

    # 使用实际尺寸导出（不要放大 pagewidth 等）
    canvas.postscript(file=ps_filename, colormode='color')

    try:
        # 打开 EPS 图像
        img = Image.open(ps_filename)
        img.load()

        # 计算新尺寸（高分辨率放大）
        new_size = (img.width * scale, img.height * scale)
        img = img.resize(new_size, resample=Image.LANCZOS)

        # 转为 PDF
        img.save(filename, "PDF")
        print(f"Saved high-res PDF to {filename}")
    except Exception as e:
        print("Failed to convert canvas to PDF:", e)
    finally:
        if os.path.exists(ps_filename):
            os.remove(ps_filename)

class SchedulingPainter:
    """Scheduling Painter"""

    def __init__(self, config: dict) -> None:
        self._device_size   = config["device_num"]
        self._devices       = config["devices"]
        self._pp_size       = config["stage_num"]
        self._pp_height     = config["pp_height"]
        self._pp_align      = config["pp_align"]
        self._pixel_base    = config["pixel_base"]
        self._max_time      = config["max_time"] if 'max_time' in config else -1
        
        self._num_microbatches = config["nmb"]

        self._basic_forward_length = [_len for _len in config["forward_length"]]
        self._basic_backward_b_length = [_len for _len in config["backward_length"]]
        self._basic_backward_w_length = [_len for _len in config["backward_length2"]]
        self._comm_length = [_len for _len in config["comm_length"]]

        self._forward_length = [_len * config["pixel_base"] for _len in config["forward_length"]]
        self._backward_b_length = [_len * config["pixel_base"] for _len in config["backward_length"]]
        self._backward_w_length = [_len * config["pixel_base"] for _len in config["backward_length2"]]

        self._tk_root = tk.Tk()
        self._tk_root.title("SchedulingPainter")

        self._highlight_state = {}
        self._item2color = {}
        self._item2block = {}
        self._item2step = {}
        self._item2mid = {}

    def _highlight_and_resume_block(self, canvas, item_id):
        if self._highlight_state[item_id]:
            self._highlight_state[item_id] = False
            canvas.itemconfig(item_id, fill=self._item2color[item_id])
        else:
            self._highlight_state[item_id] = True
            canvas.itemconfig(item_id, fill="yellow")

    def _pid2did(self, pid):
        for did in range(len(self._devices)):
            if pid in self._devices[did]:
                return did

    def generate_step_within_device(self, data:dict):
        from collections import defaultdict
        grouped = defaultdict(list)
        for key, val in data.items():
            did = key.split('_')[-1]
            grouped[did].append((key, val))

        data_step_idx = {}
        for did, kv_list in grouped.items():
            sorted_list = sorted(kv_list, key=lambda x: x[1])
            for idx, (key, _) in enumerate(sorted_list):
                data_step_idx[key] = idx
        return data_step_idx

    def draw(self, data: dict) -> None:
        """draw with tkinter"""

        # Convert data offset to pixels
        data = {key: val * self._pixel_base for key, val in data.items()}
        data_step_idx = self.generate_step_within_device(data)

        canvas_width = -1
        for k,v in data.items():
            kid, pid, mid, did = parse_microbatch_key(k)
            length = 0
            if kid == 'f':
                length = self._forward_length[pid]
            elif kid == 'b':
                length = self._backward_b_length[pid]
            elif kid == 'w':
                length = self._backward_w_length[pid]
            else:
                print("Type not found!")
            if data[k] + length + 2 * self._pp_align > canvas_width:
                max_key = k
                canvas_width = data[k] + length + 2 * self._pp_align

        _, max_key_pid, _, _ = parse_microbatch_key(max_key)

        # canvas_width = data[k] + self._backward_b_length[max_key_pid] + 2 * self._pp_align
        # 按照 Device 画示意图
        canvas_height = (self._pp_height + self._pp_align) * self._device_size

        # 0. Create label canvas
        label_canvas = tk.Canvas(self._tk_root, width=canvas_width, height=30)
        y_label = (0 + 30) // 2 + 5

        if self._max_time == -1:
            if SPLIT_BACKPROP:
                self._max_time = (data[max_key] + self._backward_w_length[max_key_pid])//self._pixel_base
            else:
                self._max_time = (data[max_key] + self._backward_b_length[max_key_pid])//self._pixel_base

        label_canvas.create_text(self._pp_align + 145, y_label, text="MinExeTime:{}, Chunk:{}, F:{}, B:{}, W:{}, C:{}".format(
                # (data[max_key] + self._backward_w_length[max_key_pid])//self._pixel_base, 
                round(self._max_time),
                self._pp_size // self._device_size,
                self._basic_forward_length[max_key_pid], 
                self._basic_backward_b_length[max_key_pid], 
                self._basic_backward_w_length[max_key_pid] if SPLIT_BACKPROP else 0, 
                # int(sum(self._comm_length) / len(self._comm_length))
                COMM_TIME
            ),
        )

        coords_label_1 = label_canvas.create_text(
            canvas_width * 0.15, y_label, text="BlockCoords:(start,end)"
        )
        coords_label_2 = label_canvas.create_text(
            canvas_width * 0.35, y_label, text="BlockCoords:(start,end)"
        )
        coords_label_3 = label_canvas.create_text(
            canvas_width * 0.55, y_label, text="BlockCoords:(start,end)"
        )

        coords_label = label_canvas.create_text(
            canvas_width - self._pp_align - 120, y_label, text="BlockCoords:(start,end)"
        )
        label_canvas.pack()

        # 1. Create main canvas
        main_canvas = tk.Canvas(self._tk_root, bg='#FFFFFF', width=canvas_width, height=canvas_height+5)
        main_canvas.pack()

        # width_scale = 100 / max(canvas_width, 1)
        # fig, ax = plt.subplots(figsize=(12 * width_scale,6))
        # _pp_align = 0
        # _pp_height = 10
        # # 画每个 device 的 timeline 背景
        # for pid in range(self._device_size):
        #     y0 = (_pp_height + _pp_align) * pid + 5
        #     y1 = y0 + _pp_height
        #     rect = patches.Rectangle((_pp_align, y0), canvas_width-2*self._pp_align, _pp_height,
        #                             linewidth=0.25, edgecolor='black', facecolor='white')
        #     ax.add_patch(rect)

        # # 画每个 microbatch block
        # for microbatch_key, offset in data.items():
        #     k, pid, mid, did = parse_microbatch_key(microbatch_key)

        #     x0 = _pp_align + offset
        #     y0 = (_pp_height + _pp_align) * did + 5

        #     if k == 'f' or k == 'r':
        #         width = self._forward_length[pid]
        #     elif k == 'b':
        #         width = self._backward_b_length[pid]
        #     elif k == 'w':
        #         width = self._backward_w_length[pid]
        #     else:
        #         width = 10

        #     x1 = x0 + width
        #     y1 = y0 + _pp_height

        #     color = set_color(pid, k, self._device_size)
        #     block = patches.Rectangle((x0, y0), width, _pp_height,
        #                             linewidth=0.25, edgecolor='black', facecolor=color)
        #     ax.add_patch(block)

        #     if SHOW_WORKLOAD_TEXT:
        #         ax.text(x0 + width / 2, y0 + _pp_height / 2,
        #                 str(mid),
        #                 ha='center', va='center', fontsize=8)

        # # 图形美化
        # ax.set_xlim(0, canvas_width)
        # ax.set_ylim(0, canvas_height)
        # ax.axis('off')
        # ax.invert_yaxis()


        # # 保存为 PDF
        # with PdfPages(f'/Users/hanayukino/pipeline_schedule_{SCHEDULE_METHOD.name}.pdf') as pdf:
        #     pdf.savefig(fig, pad_inches=0)

        # plt.show()

        # plt.close(fig)
        # print("Saved to pipeline_schedule.pdf")
        
        # return
        
        # 2. Add timeline for each pipeline
        # for pid in range(self._pp_size):
        # 按照 Device 画示意图
        pad = 0
        for pid in range(self._device_size):
            x0 = self._pp_align
            y0 = (self._pp_height + self._pp_align) * pid + pad + 5
            x1 = canvas_width - self._pp_align
            y1 = (self._pp_height + self._pp_align) * (pid + 1) - pad + 5
            main_canvas.create_rectangle(x0, y0, x1, y1, fill="#FFFFFF", outline="black")

        # 3. Draw execution block for each microbatch according to start and end time
        schedule_res_content = ""
        for microbatch_key, offset in data.items():
            k, pid, mid, did = parse_microbatch_key(microbatch_key)

            x0 = self._pp_align + offset
            # did = self._pid2did(pid=pid) # 获取对应的device id，把每个stage画在对应的device上
            # y0 = (self._pp_height + self._pp_align) * pid + pad
            y0 = (self._pp_height + self._pp_align) * did + pad + 5
            #修改画图中每个block的宽度
            block_width = self._forward_length[pid] if k in ('f', 'r') else (self._backward_b_length[pid] if k == 'b' else self._backward_w_length[pid])
            x1 = x0 + block_width
            # y1 = (self._pp_height + self._pp_align) * (pid + 1) - pad
            y1 = (self._pp_height + self._pp_align) * (did + 1) - pad + 5

            # save schedule representation in painter
            if HEAD_DP:
                schedule_res_content += "{}_{}_{}_{},{},{}\n".format(k,mid,pid,did,offset,offset+block_width)
            else:
                schedule_res_content += "{}_{}_{},{},{}\n".format(k,mid,pid,offset,offset+block_width)

            tag = f"p_{pid}_m_{mid}_{k}"
            color = set_color(pid,workload_type=k,layer_num=self._pp_size)

            block = main_canvas.create_rectangle(x0, y0, x1, y1, fill=color, tags=tag)
            # 求余考虑virtual stage的情况
            bold_font = font.Font(
                # family="Calibri Light", 
                underline= pid // self._device_size % 2,
                weight= tk.font.NORMAL if pid // self._device_size % 2 else tk.font.BOLD
            )
            if SHOW_WORKLOAD_TEXT:
                text = main_canvas.create_text(
                    (x0 + x1) // 2, (y0 + y1) // 2, text=f"{mid % self._num_microbatches}", font=bold_font
                )
                self._item2block[text] = block
            # else:
            #     if mid in (0, self._device_size + 1):
            #         text = main_canvas.create_text(
            #             (x0 + x1) // 2, (y0 + y1) // 2, text=f"{mid % self._num_microbatches}", font=bold_font
            #         )
            #         self._item2block[text] = block

            self._highlight_state[block] = False
            self._item2color[block] = color
            self._item2block[block] = block
            self._item2step[block] = data_step_idx[microbatch_key]
            # 求余考虑virtual stage的情况
            self._item2mid[block] = mid
        
        save_to_file(SCH_FILE_PATH, schedule_res_content, 'w')
        save_to_file(TEMP_RES_PATH, schedule_res_content, 'w')

        # Register hook for highlighting execution block of this microbatch
        def _trigger_hook(event):
            del event

            items = main_canvas.find_withtag("current")
            if len(items) == 0:
                return

            current_item = self._item2block[items[0]]
            if current_item not in self._highlight_state:
                return

            item_coords = main_canvas.coords(current_item)
            current_start = int(item_coords[0] - self._pp_align) // self._pixel_base
            current_end = int(item_coords[2] - self._pp_align) // self._pixel_base

            step = self._item2step[current_item]
            label_canvas.itemconfig(
                coords_label, text=f"Step {step} ({current_start},{current_end})"
            )
            label_canvas.itemconfig(
                coords_label_1, text=f"Step {step} ({current_start},{current_end})"
            )
            label_canvas.itemconfig(
                coords_label_2, text=f"Step {step} ({current_start},{current_end})"
            )
            label_canvas.itemconfig(
                coords_label_3, text=f"Step {step} ({current_start},{current_end})"
            )

            tags = [
                f"p_{pid}_m_{self._item2mid[current_item]}_{fb}"
                for pid in range(self._pp_size)
                for fb in ("f", "b", "w", "r") #点击后的效果，加上w的判断
            ]
            
            items_same_microbatch = []
            for tag in tags:
                found = main_canvas.find_withtag(tag)
                if len(found) != 0:
                    items_same_microbatch.append(found[0])

            for item in items_same_microbatch:
                self._highlight_and_resume_block(main_canvas, item)

        main_canvas.bind("<Button-1>", _trigger_hook)

        button = tk.Button(self._tk_root, text="Save as PDF", command=lambda: save_canvas_as_pdf(main_canvas))
        button.pack()
        self._tk_root.mainloop()

class MultiPipelinePainter:
    """Scheduling Painter"""

    def __init__(self, config: dict) -> None:
        self.dp_size = len(config)
        self.all_dp_config = config
        self._tk_root = tk.Tk()
        self._tk_root.title("SchedulingPainter")
        self._highlight_state = {}
        self._item2color = {}
        self._item2block = {}
        self._item2step = {}
        self._item2mid = {}

    def set_para_by_dp_idx(self, config):
        self._device_size   = config["device_num"]
        self._devices       = config["devices"]
        self._pp_size       = config["stage_num"]
        self._pp_height     = config["pp_height"]
        self._pp_align      = config["pp_align"]
        self._pixel_base    = config["pixel_base"]
        self._max_time      = config["max_time"] if 'max_time' in config else -1
        
        self._num_microbatches = config["nmb"]
        self.mid_offset = config["mid_offset"]

        self._basic_forward_length = [_len for _len in config["forward_length"]]
        self._basic_backward_b_length = [_len for _len in config["backward_length"]]
        self._basic_backward_w_length = [_len for _len in config["backward_length2"]]
        self._comm_length = [_len for _len in config["comm_length"]]

        self._forward_length = [_len * config["pixel_base"] for _len in config["forward_length"]]
        self._backward_b_length = [_len * config["pixel_base"] for _len in config["backward_length"]]
        self._backward_w_length = [_len * config["pixel_base"] for _len in config["backward_length2"]]

    def _highlight_and_resume_block(self, canvas, item_id):
        if self._highlight_state[item_id]:
            self._highlight_state[item_id] = False
            canvas.itemconfig(item_id, fill=self._item2color[item_id])
        else:
            self._highlight_state[item_id] = True
            canvas.itemconfig(item_id, fill="yellow")

    def _pid2did(self, pid):
        for did in range(len(self._devices)):
            if pid in self._devices[did]:
                return did

    def generate_step_within_device(self, data:dict):
        from collections import defaultdict
        grouped = defaultdict(list)
        for key, val in data.items():
            did = key.split('_')[-1]
            grouped[did].append((key, val))

        data_step_idx = {}
        for did, kv_list in grouped.items():
            sorted_list = sorted(kv_list, key=lambda x: x[1])
            for idx, (key, _) in enumerate(sorted_list):
                data_step_idx[key] = idx
        return data_step_idx

    def draw(self, all_dp_data: dict) -> None:
        """draw with tkinter"""
        # find longest time
        for dp_idx, data in all_dp_data.items():
            self.set_para_by_dp_idx(config=self.all_dp_config[dp_idx])
            data = {key: val * self._pixel_base for key, val in data.items()}
            canvas_width = -1
            for k,v in data.items():
                kid, pid, mid, did = parse_microbatch_key(k)
                length = 0
                if kid == 'f':
                    length = self._forward_length[pid]
                elif kid == 'b':
                    length = self._backward_b_length[pid]
                elif kid == 'w':
                    length = self._backward_w_length[pid]
                else:
                    print("Type not found!")
                if data[k] + length + 2 * self._pp_align > canvas_width:
                    max_key = k
                    canvas_width = data[k] + length + 2 * self._pp_align

        canvas_height = (self._pp_height + self._pp_align) * self._device_size * len(all_dp_data)
        # 1. Create main canvas
        main_canvas = tk.Canvas(self._tk_root, bg='#FFFFFF', width=canvas_width, height=canvas_height+5)
        main_canvas.pack()
        
        for dp_idx, data in all_dp_data.items():
            self.set_para_by_dp_idx(config=self.all_dp_config[dp_idx])
            # Convert data offset to pixels
            data = {key: val * self._pixel_base for key, val in data.items()}
            data_step_idx = self.generate_step_within_device(data)

            canvas_width = -1
            for k,v in data.items():
                kid, pid, mid, did = parse_microbatch_key(k)
                length = 0
                if kid == 'f':
                    length = self._forward_length[pid]
                elif kid == 'b':
                    length = self._backward_b_length[pid]
                elif kid == 'w':
                    length = self._backward_w_length[pid]
                else:
                    print("Type not found!")
                if data[k] + length + 2 * self._pp_align > canvas_width:
                    max_key = k
                    canvas_width = data[k] + length + 2 * self._pp_align

            _, max_key_pid, _, _ = parse_microbatch_key(max_key)

            # 0. Create label canvas
            if dp_idx == 0:
                label_canvas = tk.Canvas(self._tk_root, width=canvas_width, height=30)
                y_label = (0 + 30) // 2 + 5

                if self._max_time == -1:
                    if SPLIT_BACKPROP:
                        self._max_time = (data[max_key] + self._backward_w_length[max_key_pid])//self._pixel_base
                    else:
                        self._max_time = (data[max_key] + self._backward_b_length[max_key_pid])//self._pixel_base

                label_canvas.create_text(self._pp_align + 145, y_label, text="MinExeTime:{}, Chunk:{}, F:{}, B:{}, W:{}, C:{}".format(
                        # (data[max_key] + self._backward_w_length[max_key_pid])//self._pixel_base, 
                        round(self._max_time),
                        self._pp_size // self._device_size,
                        self._basic_forward_length[max_key_pid], 
                        self._basic_backward_b_length[max_key_pid], 
                        self._basic_backward_w_length[max_key_pid] if SPLIT_BACKPROP else 0, 
                        # int(sum(self._comm_length) / len(self._comm_length))
                        COMM_TIME
                    ),
                )

                coords_label_1 = label_canvas.create_text(
                    canvas_width * 0.15, y_label, text="BlockCoords:(start,end)"
                )
                coords_label_2 = label_canvas.create_text(
                    canvas_width * 0.35, y_label, text="BlockCoords:(start,end)"
                )
                coords_label_3 = label_canvas.create_text(
                    canvas_width * 0.55, y_label, text="BlockCoords:(start,end)"
                )

                coords_label = label_canvas.create_text(
                    canvas_width - self._pp_align - 120, y_label, text="BlockCoords:(start,end)"
                )
                label_canvas.pack()

            # 2. Add timeline for each pipeline
            # 按照 Device 画示意图
            pad = 0
            for pid in range(dp_idx * self._device_size, dp_idx * self._device_size + self._device_size):
                x0 = self._pp_align
                y0 = (self._pp_height + self._pp_align) * pid + pad + 5
                x1 = canvas_width - self._pp_align
                y1 = (self._pp_height + self._pp_align) * (pid + 1) - pad + 5
                main_canvas.create_rectangle(x0, y0, x1, y1, fill="#FFFFFF", outline="black")

            # 3. Draw execution block for each microbatch according to start and end time
            schedule_res_content = ""
            for microbatch_key, offset in data.items():
                k, pid, mid, did = parse_microbatch_key(microbatch_key)

                x0 = self._pp_align + offset
                y0 = (self._pp_height + self._pp_align) * (did + dp_idx * self._device_size) + pad + 5
                block_width = self._forward_length[pid] if k in ('f', 'r') else (self._backward_b_length[pid] if k == 'b' else self._backward_w_length[pid])
                x1 = x0 + block_width
                y1 = (self._pp_height + self._pp_align) * (did + dp_idx * self._device_size + 1) - pad + 5

                # save schedule representation in painter
                if HEAD_DP:
                    schedule_res_content += "{}_{}_{}_{},{},{}\n".format(k,mid,pid,did,offset,offset+block_width)
                else:
                    schedule_res_content += "{}_{}_{},{},{}\n".format(k,mid,pid,offset,offset+block_width)

                tag = f"p_{pid}_m_{mid}_{k}"
                color = set_color(pid,workload_type=k,layer_num=self._pp_size)

                block = main_canvas.create_rectangle(x0, y0, x1, y1, fill=color, tags=tag)
                # 求余考虑virtual stage的情况
                bold_font = font.Font(
                    # family="Calibri Light", 
                    underline= pid // self._device_size % 2,
                    weight= tk.font.NORMAL if pid // self._device_size % 2 else tk.font.BOLD
                )
                if SHOW_WORKLOAD_TEXT:
                    text = main_canvas.create_text(
                        (x0 + x1) // 2, (y0 + y1) // 2, text=f"{mid}", font=bold_font
                    )
                    self._item2block[text] = block

                self._highlight_state[block] = False
                self._item2color[block] = color
                self._item2block[block] = block
                self._item2step[block] = data_step_idx[microbatch_key]
                # 求余考虑virtual stage的情况
                self._item2mid[block] = mid

        # Register hook for highlighting execution block of this microbatch
        def _trigger_hook(event):
            del event

            items = main_canvas.find_withtag("current")
            if len(items) == 0:
                return

            current_item = self._item2block[items[0]]
            if current_item not in self._highlight_state:
                return

            item_coords = main_canvas.coords(current_item)
            current_start = int(item_coords[0] - self._pp_align) // self._pixel_base
            current_end = int(item_coords[2] - self._pp_align) // self._pixel_base

            step = self._item2step[current_item]
            label_canvas.itemconfig(
                coords_label, text=f"Step {step} ({current_start},{current_end})"
            )
            label_canvas.itemconfig(
                coords_label_1, text=f"Step {step} ({current_start},{current_end})"
            )
            label_canvas.itemconfig(
                coords_label_2, text=f"Step {step} ({current_start},{current_end})"
            )
            label_canvas.itemconfig(
                coords_label_3, text=f"Step {step} ({current_start},{current_end})"
            )

            tags = [
                f"p_{pid}_m_{self._item2mid[current_item]}_{fb}"
                for pid in range(self._pp_size)
                for fb in ("f", "b", "w", "r") #点击后的效果，加上w的判断
            ]
            
            items_same_microbatch = []
            for tag in tags:
                found = main_canvas.find_withtag(tag)
                if len(found) != 0:
                    items_same_microbatch.append(found[0])

            for item in items_same_microbatch:
                self._highlight_and_resume_block(main_canvas, item)

        main_canvas.bind("<Button-1>", _trigger_hook)

        button = tk.Button(self._tk_root, text="Save as PDF", command=lambda: save_canvas_as_pdf(main_canvas))
        button.pack()
        self._tk_root.mainloop()