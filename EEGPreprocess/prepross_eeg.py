import numpy as np
import argparse

parser = argparse.ArgumentParser(description='EEG Preprocessing')
parser.add_argument('-avg', '--average', help='Number of averages', default=80)
args = parser.parse_args()
average = int(args.average)
if average != 80:
    param = f'{average}'
else:
    param = ''

for sub in range(1, 11):
    data_dir = f'sub-{sub:02d}/'
    
    # 处理训练数据
    eeg_data_train = np.load(data_dir + 'preprocessed_eeg_training.npy', allow_pickle=True).item()
    print(f'\nTraining EEG data shape for sub-{sub:02d}:')
    print(eeg_data_train['preprocessed_eeg_data'].shape)
    print('(Training image conditions × Training EEG repetitions × EEG channels × EEG time points)')
    
    # 保留所有100个时间步并取平均
    train_thingseeg2_avg = eeg_data_train['preprocessed_eeg_data'].mean(1)  # 对重复次数维度取平均
    np.save(data_dir + 'train_thingseeg2_avg.npy', train_thingseeg2_avg)
    
    # 处理测试数据
    eeg_data_test = np.load(data_dir + 'preprocessed_eeg_test.npy', allow_pickle=True).item()
    print(f'\nTest EEG data shape for sub-{sub:02d}:')
    print(eeg_data_test['preprocessed_eeg_data'].shape)
    print('(Test image conditions × Test EEG repetitions × EEG channels × EEG time points)')
    
    # 保留所有100个时间步，取指定数量的平均
    test_thingseeg2_avg = eeg_data_test['preprocessed_eeg_data'][:,:average].mean(1)
    np.save(data_dir + f'test_thingseeg2_avg{param}.npy', test_thingseeg2_avg)
    
    print(f"Sub-{sub:02d} processed: kept all 100 time points and averaged over repetitions")