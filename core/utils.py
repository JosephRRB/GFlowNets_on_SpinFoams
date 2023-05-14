import tensorflow as tf

def print_and_log(message, file_path):
    tf.print(message)
    if file_path is not None:
        with open(file_path, "a") as f:
            f.write(message + "\n")