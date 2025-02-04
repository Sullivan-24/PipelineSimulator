"""
painter package
"""
import tkinter as tk
from tkinter import font
from .utils import parse_microbatch_key, print_to_file
from .abstract.mutils import COMM_TIME, SPLIT_BACKPROP, MICRO_BATCH_NUM, DEVICE_NUM
class SchedulingPainter:
    """Scheduling Painter"""

    def __init__(self, config: dict) -> None:
        self._device_size   = config["device_size"]
        self._devices       = config["devices"]
        self._pp_size       = config["pp_size"]
        self._pp_height     = config["pp_height"]
        self._pp_align      = config["pp_align"]
        self._pixel_base    = config["pixel_base"]
        self._max_time      = config["max_time"] if 'max_time' in config else -1
        
        self._num_microbatches = config["num_microbatches"]

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

    def draw(self, data: dict) -> None:
        """draw with tkinter"""

        # Convert data offset to pixels
        data = {key: val * self._pixel_base for key, val in data.items()}

        max_key = max(data, key=data.get)
        _, max_key_pid, _ = parse_microbatch_key(max_key)

        canvas_width = data[max_key] + self._backward_b_length[max_key_pid] + 2 * self._pp_align
        # canvas_height = (self._pp_height + self._pp_align) * self._pp_size
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

        label_canvas.create_text(self._pp_align + 160, y_label, text="MinExeTime:{}, Chunk:{}, F:{}, B:{}, W:{}, C:{}".format(
                # (data[max_key] + self._backward_w_length[max_key_pid])//self._pixel_base, 
                round(self._max_time),
                self._pp_size // self._device_size,
                self._basic_forward_length[max_key_pid], 
                self._basic_backward_b_length[max_key_pid], 
                self._basic_backward_w_length[max_key_pid], 
                # int(sum(self._comm_length) / len(self._comm_length))
                COMM_TIME
            ),
        )

        label_canvas.create_text(
            canvas_width - self._pp_align - 120, y_label, text="BlockCoords:"
        )
        coords_label = label_canvas.create_text(
            canvas_width - self._pp_align - 40, y_label, text="(start,end)"
        )
        label_canvas.pack()

        # 1. Create main canvas
        main_canvas = tk.Canvas(self._tk_root, width=canvas_width, height=canvas_height)
        main_canvas.pack()

        # 2. Add timeline for each pipeline
        # for pid in range(self._pp_size):
        # 按照 Device 画示意图
        for pid in range(self._device_size):
            x0 = self._pp_align
            y0 = (self._pp_height + self._pp_align) * pid + 5
            x1 = canvas_width - self._pp_align
            y1 = (self._pp_height + self._pp_align) * (pid + 1) - 5
            main_canvas.create_rectangle(x0, y0, x1, y1, outline="black")

        # 3. Draw execution block for each microbatch according to start and end time
        for microbatch_key, offset in data.items():
            k, pid, mid = parse_microbatch_key(microbatch_key)

            x0 = self._pp_align + offset
            did = self._pid2did(pid=pid) # 获取对应的device id，把每个stage画在对应的device上
            # y0 = (self._pp_height + self._pp_align) * pid + 5
            y0 = (self._pp_height + self._pp_align) * did + 5
            #修改画图中每个block的宽度
            block_width = self._forward_length[pid] if k == 'f' else (self._backward_b_length[pid] if k == 'b' else self._backward_w_length[pid])
            x1 = x0 + block_width
            # y1 = (self._pp_height + self._pp_align) * (pid + 1) - 5
            y1 = (self._pp_height + self._pp_align) * (did + 1) - 5

            print_to_file(f"gurobi_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}.txt", "{}_{}_{},{},{}\n".format(k,mid,pid,offset,offset+block_width))

            tag = f"p_{pid}_m_{mid}_{k}"
            if k == 'f':    #颜色设置，加上w的情况
                color = "#00AFFF"
            elif k == 'b':
                color = "#00FFFF" 
            else:
                color = "#00FF6F"

            block = main_canvas.create_rectangle(x0, y0, x1, y1, fill=color, tags=tag)
            # 求余考虑virtual stage的情况
            bold_font = font.Font(
                # family="Calibri Light", 
                underline= pid // self._device_size % 2,
                weight= tk.font.NORMAL if pid // self._device_size % 2 else tk.font.BOLD
            )
            text = main_canvas.create_text(
                (x0 + x1) // 2, (y0 + y1) // 2, text=f"{mid % self._num_microbatches}", font=bold_font
            )
            # if pid + 1 >= self._num_microbatches:
            #     bold_font = font.Font(size= pid // (self._pp_size // self._device_size), weight=tk.font.BOLD)
            #     text = main_canvas.create_text(
            #         (x0 + x1) // 2, (y0 + y1) // 2, text=f"{mid % self._num_microbatches}", font=bold_font
            #     )
            # else:
            #     text = main_canvas.create_text(
            #         (x0 + x1) // 2, (y0 + y1) // 2, text=f"{mid}"
            #     )
            # print(f"block {tag}: {x0}, {y0}, {x1}, {y1}", flush=True)

            self._highlight_state[block] = False
            self._item2color[block] = color
            self._item2block[block] = block
            self._item2block[text] = block
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
            label_canvas.itemconfig(
                coords_label, text=f"({current_start},{current_end})"
            )

            tags = [
                f"p_{pid}_m_{self._item2mid[current_item]}_{fb}"
                for pid in range(self._pp_size)
                for fb in ("f", "b", "w") #点击后的效果，加上w的判断
            ]
            
            items_same_microbatch = []
            for tag in tags:
                found = main_canvas.find_withtag(tag)
                if len(found) != 0:
                    items_same_microbatch.append(found[0])

            for item in items_same_microbatch:
                self._highlight_and_resume_block(main_canvas, item)

        main_canvas.bind("<Button-1>", _trigger_hook)

        self._tk_root.mainloop()
