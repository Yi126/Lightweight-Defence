import os
print(os.getcwd())
import sys
sys.path.append('/Project/Yi/defense')
from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Model.inception import *
from CODE.Utils.augmentation import *
from CODE.attack.methods import *
from CODE.Model.models import *

class EnsembleModel(nn.Module):
    def __init__(self, models, device, dataset_name):
        super(EnsembleModel, self).__init__()
        self.models = models  # models should be a dictionary
        self.device = device
        self.dataset_name = dataset_name
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        average_output = sum(self.models[name](inputs) for name in self.models) / len(self.models)
        #output = self.softmax(average_output)
        return average_output.requires_grad_(True)
    def print_dataset_name(self):
        print(f"Using ensemble model for dataset: {self.dataset_name}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义结果存储列表
csv_file_path = '/Project/Yi/defense/OUTPUT/ensemble2/pgd_resultensembleres.csv'

# 检查文件是否已存在
file_exists = False

try:
    with open(csv_file_path, 'r'):
        file_exists = True
except FileNotFoundError:
    file_exists = False

# 设置初始 DataFrame 列名
columns = ['Dataset Name', 'Robust Accuracy']

# 写入表头（如果文件不存在）
if not file_exists:
    results_df = pd.DataFrame(columns=columns)
    results_df.to_csv(csv_file_path, mode='w', index=False)
    file_exists = True  # 更新 file_exists，因为现在文件已经存在了
# 预加载所有模型的路径
model_paths = {
    'gn': '/Project/Yi/defense/OUTPUT/res_gn/{dataset_name}_1_model_weights.pth',
    'j': '/Project/Yi/defense/OUTPUT/res_jitter/{dataset_name}_1_model_weights.pth',
    'rz': '/Project/Yi/defense/OUTPUT/res_rz/{dataset_name}_1_model_weights.pth',
    'sz': '/Project/Yi/defense/OUTPUT/res_sz/{dataset_name}_1_model_weights.pth',
    'sts': '/Project/Yi/defense/OUTPUT/res_sts/{dataset_name}_1_model_weights.pth',
    '0': '/Project/Yi/defense/OUTPUT/resnet/{dataset_name}_0_model_weights.pth'
}

for dataset_name in UNIVARIATE_DATASET_NAMES2:
    print(f"Loading models for dataset: {dataset_name}")
    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=64)

    # 加载模型实例和预训练权重
    models = {
        "gn": ClassifierResNet18_with_gaussian_noise(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "j": ClassifierResNet18_with_Jitter(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "rz": ClassifierResNet18_with_RandomZero(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "sz": ClassifierResNet18_with_SegmentZero(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "sts": ClassifierResNet18_with_smooth_time_series(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "0": ClassifierResNet18(input_shape=1, nb_classes=ucr_nb_classes).to(device)
    }

    # 加载各个模型的权重
    for model_key in models:
        models[model_key].load_state_dict(torch.load(model_paths[model_key].format(dataset_name=dataset_name)))

    for model in models.values():
        model.eval()

    # Initialize the ensemble model
    ensemble_model = EnsembleModel(models, device, dataset_name)
    
    # Define the attacker using the ensemble model
    attack = PGD(ensemble_model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=device)

    print(f"Evaluating dataset: {dataset_name}")
    total_batches = len(test_loader)  # 获取总共的批次数
    batch_counter = 1  # 初始化计数器
    #total = 0
    #successful_attacks = 0
    #l2_distances = []
    all_labels = []
    all_predictions = []
    for inputs, labels in test_loader:
        # 将数据和标签转移到正确的设备上（例如 'cuda'）
        inputs, labels = inputs.to(device), labels.to(device)
        # 设置所有模型为评估模式
        #for model in models.values():
            #model.train()
        # 对自然样本进行预测
        #outputs_natural = model(inputs)
        #preds_natural = torch.argmax(outputs_natural, dim=1)
        # 生成对抗样本
        # 注意：在生成对抗样本时不需要 torch.no_grad()
        adv_data = attack.generate(inputs)
        #for model in models.values():
            #model.eval()
        # 对对抗样本进行预测
        outputs_adv = ensemble_model(adv_data)
        _, predicted = torch.max(outputs_adv.data, 1)
        # 将标签和预测添加到列表中
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        '''
        preds_adv = torch.argmax(outputs_adv, dim=1)
        
        # 判断攻击是否成功：对抗样本的预测与原始标签不同
        attack_success = preds_adv != labels
        #print(attack_success)
        # 累计成功的攻击
        successful_attacks += attack_success.sum().item()
        #print(successful_attacks)
        # 计算L2距离，仅对成功的攻击样本
        successful_adv_data = adv_data[attack_success]
        
        if successful_adv_data.nelement() > 0:
            successful_inputs = inputs[attack_success]

            l2_distance = torch.norm(successful_adv_data - successful_inputs, p=2, dim=[1, 2])
 
            #print(l2_distance)
            total_l2_distance = torch.sum(l2_distance).item()
            #print('df,',total_l2_distance)
            l2_distances.append(total_l2_distance)  # 将Tensor转换为列表并添加到l2_distances中
            #print(l2_distances)
            total += labels.size(0)
        '''
        torch.cuda.empty_cache()
        gc.collect()
        # 打印当前是第几个批次以及总共多少个批次
        print(f"Processing batch {batch_counter} of {total_batches} for dataset {dataset_name}")
        batch_counter += 1  # 计数器加1
    #print(l2_distances)
    # 计算攻击成功率和平均L2距离
    accuracy = accuracy_score(all_labels, all_predictions)
    
    #average_l2_distance = sum(l2_distances) / successful_attacks if successful_attacks else 0
    result = [[dataset_name, accuracy]]

    # 将结果追加到 CSV 文件
    results_df = pd.DataFrame(result, columns=columns)
    results_df.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)

    print(f"Results for {dataset_name} saved to:", csv_file_path)

print("All results have been saved.")