import tkinter as tk
from PIL import Image, EpsImagePlugin
import os
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# 设置 Ghostscript 可执行路径（macOS）
EpsImagePlugin.gs_windows_binary = "/opt/homebrew/bin/gs"  # 请根据你的机器修改路径

def save_canvas_as_pdf(canvas, filename="output.pdf"):
    # 1. 导出 Canvas 为 PostScript 文件
    ps_filename = "temp_canvas_output.ps"
    canvas.postscript(file=ps_filename, colormode='color')

    try:
        # 2. 使用 Pillow 打开并保存为 PDF
        img = Image.open(ps_filename)
        img.save(filename, "PDF")
        print(f"Canvas successfully saved as {filename}")
    except Exception as e:
        print("Failed to convert canvas to PDF:", e)
    finally:
        # 3. 清理临时 .ps 文件
        if os.path.exists(ps_filename):
            os.remove(ps_filename)


# 示例 tkinter Canvas
root = tk.Tk()
canvas = tk.Canvas(root, width=400, height=300, bg='white')
canvas.pack()

# 画个东西
canvas.create_oval(50, 50, 200, 200, fill='lightblue')
canvas.create_text(125, 125, text="Hello", font=("Arial", 24))

# 保存按钮
button = tk.Button(root, text="Save as PDF", command=lambda: save_canvas_as_pdf(canvas))
button.pack()

root.mainloop()
