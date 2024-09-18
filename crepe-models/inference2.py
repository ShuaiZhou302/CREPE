import librosa
import numpy as np
import tensorflow as tf


# 加载 .pb 文件模型
def load_model(pb_file):
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


# 从 .wav 文件中提取音频数据
def load_wav_file(wav_file, sample_rate=16000):
    # librosa 加载音频文件，并将其重新采样到指定的采样率 (16kHz)
    audio, sr = librosa.load(wav_file, sr=sample_rate)
    return audio


# 将音频数据切分为 1024 个采样的帧
def create_input_data_from_audio(audio, frame_size=1024):
    # 确保音频数据长度是1024的倍数
    if len(audio) < frame_size:
        # 如果音频小于1024个样本，进行零填充
        audio = np.pad(audio, (0, frame_size - len(audio)), 'constant')
    else:
        audio = audio[:frame_size]  # 截取前1024个样本
    return audio.reshape(1, frame_size).astype(np.float32)


# 执行推理
def run_inference(graph_def, input_data):
    with tf.compat.v1.Session() as sess:
        # 将图加载到当前会话
        tf.import_graph_def(graph_def, name="")

        # 获取输入和输出的操作
        input_tensor = sess.graph.get_tensor_by_name("frames:0")
        output_tensor = sess.graph.get_tensor_by_name("model/classifier/Sigmoid:0")

        # 执行推理
        predictions = sess.run(output_tensor, feed_dict={input_tensor: input_data})
        return predictions


# 文件路径
pb_file = 'crepe-tiny-1.pb'
wav_file = 'do_re_mi_fa_so_la_xi.wav'  # 你的 .wav 文件路径

# 加载模型
graph_def = load_model(pb_file)

# 从 .wav 文件中加载音频数据
audio_data = load_wav_file(wav_file)

# 创建输入数据（长度为 1024 的帧）
input_data = create_input_data_from_audio(audio_data)

# 执行推理并获取结果
predictions = run_inference(graph_def, input_data)

# 输出结果
print("Predictions:", predictions)
