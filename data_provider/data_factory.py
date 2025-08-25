from torch.utils.data import DataLoader

'''
## 主要功能

data_factory.py 的核心作用是根据任务类型和参数，动态生成合适的数据集对象和 DataLoader，为模型训练、验证和测试阶段提供数据输入。它通过 data_provider 函数实现这一功能。

## 逻辑顺序

1. **数据集类型映射**  
   文件开头定义了一个 `data_dict` 字典，将数据集名称（如 'ETTh1', 'PSM', 'UEA' 等）映射到对应的数据集类（如 `Dataset_ETT_hour`, `PSMSegLoader` 等）。

2. **data_provider 函数入口**  
   - 输入参数：`args`（命令行参数或配置对象）、`flag`（'train'、'val'、'test'）。
   - 通过 `args.data` 查找对应的数据集类 `Data`。

3. **参数设置**  
   - 判断 `flag` 是否为 `'test'`，决定是否打乱数据（`shuffle_flag`）、是否丢弃最后一个 batch（`drop_last`）、batch 大小（`batch_size`）等。
   - 设定时间编码方式 `timeenc`。

4. **根据任务类型分支**  
   - **异常检测（anomaly_detection）**  
     - 创建异常检测数据集对象，batch size 通常为 1，drop_last=False。
     - 返回数据集和 DataLoader。
   - **分类（classification）**  
     - 创建分类数据集对象，drop_last=False，使用自定义 `collate_fn`。
     - 返回数据集和 DataLoader。
   - **其他任务（如预测、补全等）**  
     - 创建通用数据集对象，传入各种参数（如序列长度、特征、目标、时间编码、噪声等）。
     - 返回数据集和 DataLoader。

5. **返回**  
   - 每个分支都会返回 `(data_set, data_loader)`，供主程序调用。

- **data_factory.py** 是数据输入的“工厂”，根据参数自动选择和构造合适的数据集和 DataLoader。
- 你在 debug 时，通常会看到主程序调用 data_provider，然后跳转到不同的数据集类（如 `Dataset_ETT_hour`、`PSMSegLoader` 等）的初始化方法。
- 这个文件是数据流入模型的第一站，后续数据会被送入模型进行训练或推理。

如需追踪更细致的跳转（比如具体到某个数据集类的实现），可以在 data_provider 和各数据集类的 `__init__` 方法处加断点。
'''
from data_provider.data_loader import (Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_hour_Trend,
                                       Dataset_ETT_minute, Dataset_M4,
                                       MSLSegLoader, PSMSegLoader, SMAPSegLoader, SMDSegLoader,
                                       SWATSegLoader, UEAloader)
from data_provider.uea import collate_fn

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh1_Trend': Dataset_ETT_hour_Trend,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    #很奇怪，明明task_name是long_term forecast的情况下还是跳进了anomaly和classification的循环
    #但是值没有变。
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        #用一组参数创建一个Data类的对象，并赋值给变量dataset
        #关键字参数实例化
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            add_noise=args.add_noise,
            noise_amp=args.noise_amp,
            noise_freq_percentage=args.noise_freq_percentage,
            noise_seed=args.noise_seed,
            noise_type=args.noise_type,
            data_percentage=args.data_percentage,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
