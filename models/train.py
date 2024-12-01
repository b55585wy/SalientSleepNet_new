import tensorflow as tf
import yaml
from models.sleep_stage_sst import SleepStageSST  # 假设这里你也有一个类似的TensorFlow模型定义
from utils.metrics import compute_metrics  # 你可能需要调整计算metrics的方法，确保它支持TensorFlow


def train(config):
    # 初始化模型
    model = SleepStageSST(config)  # 你需要确保这个模型是TensorFlow版本

    # 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])

    # 损失函数
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 数据加载部分，使用tf.data.Dataset
    train_dataset = tf.data.Dataset.from_generator(
        train_loader,  # 你需要替换train_loader为TensorFlow兼容的生成器或tf.data.Dataset
        output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # 假设信号数据是浮点数
            tf.TensorSpec(shape=(None,), dtype=tf.int32)  # 假设标签是整数
        )
    )

    valid_dataset = tf.data.Dataset.from_generator(
        valid_loader,  # 同样的替换
        output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    # 训练循环
    for epoch in range(config['training']['epochs']):
        # 训练模式
        model.trainable = True
        for signals, labels in train_dataset:
            with tf.GradientTape() as tape:
                logits = model(signals, training=True)
                loss = criterion(labels, logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # 验证
        model.trainable = False
        metrics = evaluate(model, valid_dataset)  # 你需要确保evaluate方法适用于TensorFlow
        print(f"Epoch {epoch}: {metrics}")


def evaluate(model, valid_dataset):
    # 评估模型
    total_loss = 0
    total_metrics = []
    for signals, labels in valid_dataset:
        logits = model(signals, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        total_loss += tf.reduce_sum(loss)
        # 计算你的metrics，可以调用compute_metrics
        metrics = compute_metrics(labels, logits)
        total_metrics.append(metrics)

    # 返回总损失和计算的指标
    avg_loss = total_loss / len(valid_dataset)
    avg_metrics = tf.reduce_mean(total_metrics, axis=0)
    return {"loss": avg_loss, "metrics": avg_metrics}


if __name__ == "__main__":
    # 加载配置文件
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    train(config)
