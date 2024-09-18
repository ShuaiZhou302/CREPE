import tensorflow as tf
import numpy as np


# 加载 .pb 文件模型
def load_model(pb_file):
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


# 创建输入数据，假设是长度为 1024 的浮点数数组
def create_input_data():
    # 随机生成输入数据进行测试
    return np.random.rand(1, 1024).astype(np.float32)


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

# 加载模型
graph_def = load_model(pb_file)

# 创建输入数据
input_data = create_input_data()

# 执行推理并获取结果
predictions = run_inference(graph_def, input_data)

# 输出结果
print("Predictions:", predictions)
