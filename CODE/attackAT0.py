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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义结果存储列表
results = []

for dataset_name in UNIVARIATE_DATASET_NAMES:
    print(dataset_name)
    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=64)
    # 重新创建模型实例并加载权重
    model = Classifier_INCEPTION(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    model.load_state_dict(torch.load(f'/Project/Yi/defense/OUTPUT/AT040/{dataset_name}_AT040_model_weights.pth'))

    model.eval()  # 将模型设置为评估模式

    #attack = FGSM(model, eps=0.1, device=device)
    #attack = BIM(model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=device)
    #attack = GM(model, eps_init=0.001, eps=0.1, num_iters=1000, device=device)
    attack = SWAP(model, eps_init=0.001, eps=0.1, gamma=0.01, num_iters=1000, device=device)
    #attack = PGD(model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=device)
    #attack = CW(model, eps_init=0.001, eps=0.1, c=1e-5, num_iters=1000, device=device)
    # 初始化用来计算攻击成功率的变量
    total = 0
    successful_attacks = 0
    l2_distances = []
    # 遍历测试数据集
    all_labels = []
    all_predictions = []
    for inputs, labels in test_loader:
        # 将数据和标签转移到正确的设备上（例如 'cuda'）
        inputs, labels = inputs.to(device), labels.to(device)

        # 对自然样本进行预测
        #outputs_natural = model(inputs)
        #preds_natural = torch.argmax(outputs_natural, dim=1)
        # 生成对抗样本
        # 注意：在生成对抗样本时不需要 torch.no_grad()
        adv_data = attack.generate(inputs)
        
        # 对对抗样本进行预测
        outputs_adv = model(adv_data)
        _, predicted = torch.max(outputs_adv.data, 1)
        # 将标签和预测添加到列表中
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

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

    #print(l2_distances)
    # 计算攻击成功率和平均L2距离
    accuracy = accuracy_score(all_labels, all_predictions)
    
    average_l2_distance = sum(l2_distances) / successful_attacks if successful_attacks else 0
    
    # 添加结果到列表
    results.append([dataset_name, accuracy, average_l2_distance])

# 将结果列表转换为 DataFrame
results_df = pd.DataFrame(results, columns=['Dataset Name', 'Robust Accuracy', 'Average L2 Distance'])
# 定义 CSV 文件路径并保存结果
csv_file_path = '/Project/Yi/defense/OUTPUT/AT040/pgd_resultsAT040.csv'
results_df.to_csv(csv_file_path, index=False)

print("Results saved to:", csv_file_path)