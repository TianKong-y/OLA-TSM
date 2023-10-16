import numpy as np
import librosa
import soundfile as sf

frame_length = 0.1 #单位均为:s
Ha = 0
Hs = 0.05
ifile_name = "./origin/in.wav"
ofile_name = "./result/out.wav"


def init():
	print("使用方法:")
	print("1.将需要变换的音频放在/origin文件夹内(仅支持wav格式), 并重命名为in.wav")
	print("2.在下方按照指示输入信息")
	print("======")
	global op
	op = int(input("请输入想转化的类型(1表示变速不变调, 2表示变调不变速):"))
	while(op < 1 or op > 2):
		op = int(input("输入不合法, 请重新输入:"))
	
	global time_shift
	if(op == 2):
		global freq_shift
		freq_shift = float(input("请输入目标频率倍率:"))
		time_shift = freq_shift
	else:
		time_shift = float(input("请输入目标时长倍率(注意不是倍速, 而是时长倍率):"))

	global Ha, Hs
	Ha = Hs / time_shift

init()

y, sr = librosa.load(ifile_name, sr = None) #sr为采样率
duration = librosa.get_duration(y = y, sr = sr)
print("音频时长:"+str(duration))

#裁切音频
frames = []
print("正在裁切音频")
cnt = 0
dt_Sampling_points = int(Ha * sr)
for i in range(0, int(duration * sr), dt_Sampling_points):
	l = i
	r = i + int(frame_length * sr) - 1
	if(int(duration * sr) <= r):
		tmp = np.array(y[l:r])
		frames.append(tmp)
		break
	tmp = np.array(y[l:r])
	frames.append(tmp)
	cnt = cnt + 1 #更新帧数

#加窗
print("正在加窗")
hanwindow = np.hanning(len(frames[0]))
for i in range(0, cnt):
	frames[i] = np.multiply(frames[i], hanwindow)

#叠加
print("正在叠加")
res = np.zeros_like(frames[0])
res_length = int(Hs * (cnt - 2) * sr) + len(frames[cnt - 1])
res = np.pad(res, (0, res_length - len(frames[0])))
for i in range(0, cnt):
	for j in range(len(frames[i])):
		res[int(Hs * (i - 1) * sr) + j] += frames[i][j]

#输出
print("正在输出")
if(op == 2): #变调不变速
	sf.write(ofile_name, res, int(sr * time_shift)) #直接改变采样率(单位时间内的采样点数量)，即可改变速率
else: #变速不变调
	sf.write(ofile_name, res, sr)

print("done!")
