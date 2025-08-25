# 不指定文件路径，用户可自行调整
import re
import matplotlib.pyplot as plt

# 读取log文件
with open('/Users/rachel/CODE/FreDF/job_9285.log', 'r') as f:
    log = f.read()

# 用正则提取每组实验的参数和结果
pattern = r'Alpha:\s*([\d\.]+).*?Pred Len:\s*(\d+).*?test \d+\n(\d+)\s*\| mse:([\d\.]+), mae:([\d\.]+)'
matches = re.findall(pattern, log, re.DOTALL)

# 整理为字典列表
results = []
for m in matches:
    alpha = float(m[0])
    pred_len = int(m[2])
    mse = float(m[3])
    mae = float(m[4])
    results.append({'alpha': alpha, 'pred_len': pred_len, 'mse': mse, 'mae': mae})

# 按预测长度分组画图
pred_lens = sorted(set(r['pred_len'] for r in results))
for pl in pred_lens:
    subset = [r for r in results if r['pred_len'] == pl]
    alphas = [r['alpha'] for r in subset]
    mses = [r['mse'] for r in subset]
    maes = [r['mae'] for r in subset]
    plt.plot(alphas, mses, label=f'MSE pred_len={pl}')
    plt.plot(alphas, maes, label=f'MAE pred_len={pl}', linestyle='--')

plt.xlabel('Alpha')
plt.ylabel('Error')
plt.title('Test Error vs Alpha')
plt.legend()
plt.grid(True)
plt.show()