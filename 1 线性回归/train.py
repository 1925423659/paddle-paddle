import argparse
import sys
import math
import paddle
import paddle.fluid

def parse_args():
    # -h 的 usage 中展示的信息
    parser = argparse.ArgumentParser('fit_a_line')
    parser.add_argument(
        '--enable_ce',
        action = 'store_true',
        help = 'If set, run the task with continuous evaluation logs.')
    parser.add_argument(
        '--use_gpu',
        type = bool,
        default = False,
        help = 'Whether to use GPU or not.')
    parser.add_argument(
        '--num_epochs',
        type = int,
        default = 100,
        help = 'number of epochs.')
    args = parser.parse_args()
    return args

# 配置数据提供器（Datafeeder）
def get_reader(batch_size):
    train = paddle.dataset.uci_housing.train()
    test = paddle.dataset.uci_housing.test()
    if args.enable_ce:
        reader_train = paddle.batch(
            train,
            batch_size = batch_size)
        reader_test = paddle.batch(
            test,
            batch_size = batch_size)
    else:
        shuffle_train = paddle.reader.shuffle(
            train,
            buf_size = 500)
        shuffle_test = paddle.reader.shuffle(
            test,
            buf_size = 500)
        reader_train = paddle.batch(
            shuffle_train,
            batch_size = batch_size)
        reader_test = paddle.batch(
            shuffle_test,
            batch_size = batch_size)
    return reader_train, reader_test

# 配置训练程序
def get_fetch():
    # 定义输入的形状和数据类型
    x = paddle.fluid.layers.data(
        name = 'x',
        shape = [13],
        dtype = 'float32')
    # 定义输出的形状和数据类型
    y = paddle.fluid.layers.data(
        name = 'y',
        shape = [1],
        dtype = 'float32')
    # 连接输入和输出的全连接层
    y_predict = paddle.fluid.layers.fc(
        input = x,
        size = 1,
        act = None)
    # 利用标签数据和输出的预测数据估计方差
    cost = paddle.fluid.layers.square_error_cost(
        input = y_predict,
        label = y)
    # 对方差求均值，得到平均损失
    avg_loss = paddle.fluid.layers.mean(cost)
    return x, y, y_predict, avg_loss

# Optimizer Function 配置
def optimizer(learning_rate, avg_loss):
    # optimizer SGD、learning_rate 是学习率，与网络的训练收敛速度有关系
    optimizer_sgd = paddle.fluid.optimizer.SGD(learning_rate = learning_rate)
    optimizer_sgd.minimize(avg_loss)

# 定义运算场所
def get_executor():
    place = paddle.fluid.CUDAPlace(0) if args.use_gpu else paddle.fluid.CPUPlace()
    executor = paddle.fluid.Executor(place)
    executor_test = paddle.fluid.Executor(place)
    return place, executor, executor_test

def get_program():
    # 获取默认/全局启动程序
    program_startup = paddle.fluid.default_startup_program()
    # 获取默认/全局主函数
    program_main = paddle.fluid.default_main_program()
    # 克隆 main_program 得到 test_program
    # 有些 operator 在训练和测试之间的操作是不同的，例如 batch_norm，使用参数 for_test 来区分该程序是用来训练还是用来测试
    # 该 api 不会删除任何操作符，请在 backward 和 optimization 之前使用
    program_test = program_main.clone(for_test = True)
    if args.enable_ce:
        program_main.random_seed = 90
        program_startup.random_seed = 90
    return program_startup, program_main, program_test


def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program = program,
            feed = feeder.feed(data_test),
            fetch_list = fetch_list)
        print(outs)
        print(zip(accumulated, outs))
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]

def main():
    batch_size = 20

    reader_train, reader_test = get_reader(batch_size = batch_size)

    x, y, y_predict, avg_loss = get_fetch()

    optimizer(learning_rate = 0.001, avg_loss = avg_loss)

    place, executor, executor_test = get_executor()

    feeder = paddle.fluid.DataFeeder(
        place = place,
        feed_list = [x, y])

    program_startup, program_main, program_test = get_program()

    #### 训练主循环
    params_dirname = 'fit_a_line.inference.model'
    num_epochs = args.num_epochs
    prompt_train = 'Train cost'
    prompt_test = 'Test cost'
    step = 0
    
    executor.run(program_startup)
    for pass_id in range(num_epochs):
        for data_train in reader_train():
            avg_loss_value = executor.run(
                program = program_main,
                feed = feeder.feed(data_train),
                fetch_list = [avg_loss])
            if step % 10 == 0:
                print('%s, Step %d, Cost %f' % (prompt_train, step, avg_loss_value[0]))
            if step % 100 == 0:
                test_metics = train_test(
                    executor = executor_test,
                    program = program_test,
                    reader = reader_test,
                    fetch_list = [avg_loss],
                    feeder = feeder)
                print('%s, Step %d, Cost %f' % (prompt_test, step, test_metics[0]))
                if test_metics[0] < 10.0:
                    break
            
            step += 1

            if math.isnan(float(avg_loss_value[0])):
                sys.exit('got NaN loss, training failed.')
        if params_dirname is not None:
            paddle.fluid.io.save_inference_model(
                params_dirname,
                ['x'],
                [y_predict],
                executor)
        if args.enable_ce and pass_id == num_epochs - 1:
            print('kpis\ttrain_cost\t%f' % avg_loss_value[0])
            print('kpis\ttest_cost\t%f' % test_metics[0])
    
    # executor_inference = paddle.fluid.Executor(place)
    # scope_inference = paddle.fluid.core.Scope()

    # with paddle.fluid.scope_guard(scope_inference):
    #     [inference_program, feed_target_names, fetch_targets] = paddle.fluid.io.load_inference_model(
    #         params_dirname,
    #         executor_inference)
    #     batch_size = 10


if __name__ == '__main__':
    args = parse_args()
    main()