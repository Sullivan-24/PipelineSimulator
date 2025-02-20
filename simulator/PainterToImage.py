from PIL import ImageGrab
import os
def save_canvas_as_image(canvas, filename):
    # # 获取画布的坐标
    # x = canvas.winfo_rootx()
    # y = canvas.winfo_rooty()
    # width = canvas.winfo_width()
    # height = canvas.winfo_height()

    # # 截取画布区域的图像
    # image = ImageGrab.grab(bbox=(x, y, x + width, y + height))

    # # 保存图像
    # filepath = os.path.join('schedule_results/images/', filename)
    # image.save(filepath)
    filepath = os.path.join('schedule_results/images/', filename+'.ps')
    print(filepath)
    canvas.update()
    canvas.postscript(file=filepath, colormode='color')    #扩展名可为ps或esp
